import os
import json
import shutil
import re

FINDINGS_PATH = r"C:\DONT SKIP CLASSES\HCMUT\RESEARCH\CHEX\KAN-TRaCE-local\data\findings_fixed.json"
INPUT_JSONL = r"C:\DONT SKIP CLASSES\HCMUT\RESEARCH\CHEX\KAN-TRaCE-local\data\chexplpus_metadata_final_raw.jsonl"
OUTPUT_JSONL = INPUT_JSONL  # overwrite in place
WORKING_JSONL = INPUT_JSONL + ".working"
CHECKPOINT_JSON = INPUT_JSONL + ".checkpoint.json"
COPY_TO = r"C:\DONT SKIP CLASSES\HCMUT\RESEARCH\CHEX\KAN-TRaCE-local\data\chexplpus_metadata_final.jsonl"
SUMMARY_PATH = r"C:\DONT SKIP CLASSES\HCMUT\RESEARCH\CHEX\KAN-TRaCE-local\data\chexplus_summary.txt"
CHECKPOINT_EVERY = 5000

# target order mapping provided by user
LABEL_ORDER = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Consolidation",
    3: "Edema",
    4: "Enlarged Cardiomediastinum",
    5: "Fracture",
    6: "Lung Lesion",
    7: "Lung Opacity",
    8: "No Finding",
    9: "Pleural Effusion",
    10: "Pleural Other",
    11: "Pneumonia",
    12: "Pneumothorax",
    13: "Support Devices",
}

# reverse map for name->index
NAME_TO_IDX = {v: k for k, v in LABEL_ORDER.items()}
VIEW_TOKEN_RE = re.compile(r"view\d+_(?:frontal|lateral)", re.IGNORECASE)


def normalize_value(v):
    if v is None:
        return -100
    try:
        if float(v) == 1.0:
            return 1
        if float(v) == 0.0:
            return 0
        if float(v) == -1.0:
            return -100
    except Exception:
        pass
    return -100


def load_findings_map(path):
    mapping = {}
    dicom_index = {}
    path_map = {}
    if not os.path.exists(path):
        print('findings file not found:', path)
        return mapping, dicom_index, path_map
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            p = obj.get('path_to_image') or obj.get('path')
            if not p:
                continue
            parts = p.split('/')
            if len(parts) < 3:
                continue
            # patient dir, study dir, dicom file (lowercased for stable keys)
            patient = parts[1].lower()
            study = parts[2].lower()
            dicom = os.path.splitext(parts[-1])[0].lower()
            key = (patient, study, dicom)
            mapping[key] = obj
            if dicom:
                dicom_index.setdefault(dicom, []).append(key)
            # normalized path keys for direct matching
            norm_full = p.replace('\\', '/').lstrip('/')
            path_map[norm_full.lower()] = obj
            # also store last 3 parts (patient/study/file) and variants without extension
            tail = '/'.join(parts[-3:])
            path_map[tail.lower()] = obj
            # store basename without extension
            path_map[os.path.splitext(parts[-1])[0].lower()] = obj
            # store patient/study/dicom (dicom usually view token) for direct matching
            try:
                patient_study_view = '/'.join([parts[-3].lower(), parts[-2].lower(), os.path.splitext(parts[-1])[0].lower()])
                path_map[patient_study_view] = obj
                # also store with train prefix if present
                if parts[0].lower() == 'train':
                    path_map['train/' + patient_study_view] = obj
            except Exception:
                pass
    return mapping, dicom_index, path_map


def build_label_vector_from_obj(obj):
    vec = [-100] * 14
    for name, idx in NAME_TO_IDX.items():
        raw = obj.get(name)
        vec[idx] = normalize_value(raw)
    return vec


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_JSON):
        return 0, 0
    try:
        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return int(data.get('last_line', 0)), int(data.get('updated', 0))
    except Exception:
        return 0, 0


def save_checkpoint(last_line, updated):
    tmp_path = CHECKPOINT_JSON + '.tmp'
    payload = {
        'last_line': int(last_line),
        'updated': int(updated),
    }
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    os.replace(tmp_path, CHECKPOINT_JSON)


def remove_checkpoint():
    for path in (CHECKPOINT_JSON, CHECKPOINT_JSON + '.tmp'):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def count_lines(path):
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            total += 1
    return total


def try_match_record(record, findings_map, dicom_index, path_map):
    # obtain patient, study, dicom candidates from record
    image_path = record.get('image_path') or record.get('image') or ''
    image_id = record.get('image_id') or ''
    patient = ''
    study = ''
    dicom = ''
    # attempt to extract last three parts from image_path
    try:
        parts = image_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            patient = parts[-3]
            study = parts[-2]
            fname = parts[-1]
            dicom = os.path.splitext(fname)[0]
    except Exception:
        pass
    # candidates for matching
    candidates = []
    if patient and study and dicom:
        candidates.append((patient.lower(), study.lower(), dicom.lower()))
    # try variations: patient may have prefix 'patient' or not
    if patient.startswith('patient'):
        pnum = patient
        pnum_strip = patient.replace('patient', '')
    else:
        pnum = 'patient' + patient
        pnum_strip = patient
    if study.startswith('study'):
        snum = study
        snum_strip = study.replace('study', '')
    else:
        snum = 'study' + study
        snum_strip = study
    # candidate variants
    if pnum and snum and dicom:
        candidates.append((pnum.lower(), snum.lower(), dicom.lower()))
    if pnum_strip and snum and dicom:
        candidates.append((pnum_strip.lower(), snum.lower(), dicom.lower()))
    if pnum and snum_strip and dicom:
        candidates.append((pnum.lower(), snum_strip.lower(), dicom.lower()))
    if pnum_strip and snum_strip and dicom:
        candidates.append((pnum_strip.lower(), snum_strip.lower(), dicom.lower()))
    # also try to match dicom token inside image_id
    if image_id:
        # try basename of image_id as dicom token and lookup index (fast)
        try:
            fname = os.path.basename(image_id)
            dic_candidate = os.path.splitext(fname)[0].lower()
            if dic_candidate in dicom_index:
                for key in dicom_index[dic_candidate]:
                    candidates.append(key)
        except Exception:
            pass
        # also try extracting suffix tokens from chexplus-style basenames
        try:
            fname2 = os.path.basename(image_path)
            base = os.path.splitext(fname2)[0].lower()
            parts_tokens = base.split('_')
            # try last 2 and last 3 tokens joined
            for n in (2, 3):
                if len(parts_tokens) >= n:
                    cand = '_'.join(parts_tokens[-n:]).lower()
                    if cand in dicom_index:
                        for key in dicom_index[cand]:
                            candidates.append(key)
        except Exception:
            pass
    # iterate candidates
    # first try direct normalized path matching
    try:
        img_norm = image_path.replace('\\', '/').lstrip('/').lower()
        if img_norm in path_map:
            return path_map[img_norm]
        tail = '/'.join(img_norm.split('/')[-3:]).lower()
        if tail in path_map:
            return path_map[tail]
        base = os.path.splitext(os.path.basename(img_norm))[0].lower()
        if base in path_map:
            return path_map[base]
        # try patient/study/view token extraction from absolute-style chexplus paths
        parts_full = img_norm.split('/')
        # find token that looks like patientNNNNN
        patient_idx = None
        for i, tok in enumerate(parts_full):
            if tok.startswith('patient'):
                patient_idx = i
                break
        if patient_idx is not None and patient_idx + 1 < len(parts_full):
            patient_tok = parts_full[patient_idx]
            study_tok = parts_full[patient_idx + 1]
            view_match = VIEW_TOKEN_RE.search(img_norm)
            if view_match:
                view_token = view_match.group(0).lower()
                key1 = '/'.join([patient_tok, study_tok, view_token]).lower()
                if key1 in path_map:
                    return path_map[key1]
                key2 = 'train/' + key1
                if key2 in path_map:
                    return path_map[key2]
            fname = os.path.splitext(parts_full[-1])[0]
            fname_tokens = fname.split('_')
            if len(fname_tokens) >= 2:
                # fallback when view token is the tail of the filename
                view_token = '_'.join(fname_tokens[-2:])
                key1 = '/'.join([patient_tok, study_tok, view_token]).lower()
                if key1 in path_map:
                    return path_map[key1]
                key2 = 'train/' + key1
                if key2 in path_map:
                    return path_map[key2]
    except Exception:
        pass

    for c in candidates:
        if c in findings_map:
            return findings_map[c]
    return None


def process():
    findings_map, dicom_index, path_map = load_findings_map(FINDINGS_PATH)
    print('Loaded findings entries:', len(findings_map), 'path keys:', len(path_map))
    if not os.path.exists(INPUT_JSONL):
        print('input chexplus final not found:', INPUT_JSONL)
        return
    total_lines = count_lines(INPUT_JSONL)
    last_line, updated = load_checkpoint()
    resume_mode = last_line > 0 and os.path.exists(WORKING_JSONL)
    if resume_mode:
        print('Resuming from checkpoint line', last_line, 'updated', updated)
    else:
        print('Starting fresh run')
        last_line = 0
        updated = 0
        for path in (WORKING_JSONL, CHECKPOINT_JSON, CHECKPOINT_JSON + '.tmp'):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    total = 0
    mode = 'a' if resume_mode else 'w'
    with open(INPUT_JSONL, 'r', encoding='utf-8') as src, open(WORKING_JSONL, mode, encoding='utf-8') as dst:
        try:
            for current_line, line in enumerate(src, start=1):
                if current_line <= last_line:
                    continue
                total = current_line
                if total % 10000 == 0:
                    print('Processed', total, 'of', total_lines, 'lines...')
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                match = try_match_record(row, findings_map, dicom_index, path_map)
                if match:
                    vec = build_label_vector_from_obj(match)
                    if updated < 20:
                        print('Updating record:', row.get('image_path') or row.get('image') or row.get('image_id'))
                        print('  old labels:', row.get('labels'))
                        print('  new labels:', vec)
                        print('  matched findings path:', match.get('path_to_image') or match.get('path'))
                    row['labels'] = vec
                    updated += 1
                if not isinstance(row.get('labels'), list) or len(row.get('labels')) != 14:
                    pass
                dst.write(json.dumps(row, ensure_ascii=False) + '\n')
                if total % CHECKPOINT_EVERY == 0:
                    dst.flush()
                    os.fsync(dst.fileno())
                    save_checkpoint(total, updated)
                    print('Checkpoint saved at line', total, 'updated', updated)
            dst.flush()
            os.fsync(dst.fileno())
            save_checkpoint(total_lines, updated)
        except KeyboardInterrupt:
            dst.flush()
            os.fsync(dst.fileno())
            save_checkpoint(total, updated)
            print('Interrupted; checkpoint saved at line', total, 'updated', updated)
            return
        except Exception as e:
            dst.flush()
            os.fsync(dst.fileno())
            save_checkpoint(total, updated)
            print('Error during processing:', e)
            return

    shutil.move(WORKING_JSONL, INPUT_JSONL)
    # copy to MyChex data
    os.makedirs(os.path.dirname(COPY_TO), exist_ok=True)
    shutil.copyfile(INPUT_JSONL, COPY_TO)
    print(f'Processed {total} records, updated {updated} labels')

    # recompute prevalence and write summary
    counts = [0]*14
    total_train = 0
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('split') != 'train':
                continue
            total_train += 1
            labels = obj.get('labels') or []
            if isinstance(labels, list) and len(labels) == 14:
                for i, v in enumerate(labels):
                    if v == 1:
                        counts[i] += 1
    # write summary
    lines = []
    lines.append(f'Total train samples: {total_train}')
    lines.append('Label,Positive Count,Prevalence%')
    for i in range(14):
        pct = (counts[i]/total_train*100) if total_train>0 else 0
        lines.append(f"{LABEL_ORDER[i]},{counts[i]},{pct:.2f}")
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as sf:
        sf.write('\n'.join(lines))
    print('Wrote summary to', SUMMARY_PATH)
    remove_checkpoint()

if __name__ == '__main__':
    process()
