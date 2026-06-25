from __future__ import annotations

import argparse
import json
from pathlib import Path

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", 
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", 
    "Pneumonia", "Pneumothorax", "Support Devices"
]
NIH_TO_CHEX_MAP = {
    0: 0,   # Atelectasis
    1: 1,   # Cardiomegaly
    2: 9,  # Effusion -> Pleural Effusion
    3: 7,   # Infiltration -> Lung Opacity
    4: 6,   # Mass -> Lung Lesion
    5: 6,   # Nodule -> Lung Lesion
    6: 11,   # Pneumonia
    7: 12,   # Pneumothorax
    8: 2,   # Consolidation
    9: 3,   # Edema
    12: 10   # Pleural Thickening -> Pleural Other
}
NORMAL_REPORT = (
    "The chest X-ray is normal. "
    "No significant findings or acute cardiopulmonary abnormalities are seen."
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic NIH reports from label vectors")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata.jsonl"),
        help="Input NIH metadata JSONL path",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata.jsonl"),
        help="Output NIH metadata JSONL path with generated reports",
    )
    return parser.parse_args()


def labels_to_report(labels: list[int]) -> str:
    # Lấy tên các nhãn có giá trị dương tính (1)
    findings = [CHEXPERT_LABELS[i] for i, val in enumerate(labels) if val == 1]
    
    if not findings:
        return compose_report("", NORMAL_REPORT)

    # Sửa lại câu từ cho khớp với tên nhãn mới
    if len(findings) == 1:
        impression_text = f"The chest X-ray shows evidence of {findings[0].lower()}."
    elif len(findings) == 2:
        impression_text = f"The chest X-ray shows evidence of {findings[0].lower()} and {findings[1].lower()}."
    else:
        finding_list = ", ".join([f.lower() for f in findings[:-1]]) + f", and {findings[-1].lower()}"
        impression_text = f"The chest X-ray shows evidence of {finding_list}."

    return compose_report("", impression_text)


def compose_report(findings: str, impression: str) -> str:
    return f"FINDINGS: {findings}; IMPRESSION: {impression};"


def process_jsonl(input_jsonl: Path, output_jsonl: Path) -> tuple[int, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            labels = row.get("labels")
            if not isinstance(labels, list):
                skipped += 1
                continue

            # --- KHÚC CẦN THAY THẾ ---
            labels = row.get("labels", [])
            
            # 1. Bỏ trường bboxes và findings theo yêu cầu
            row.pop("bboxes", None)
            row.pop("findings", None)
            row.pop("view", None) # Xóa luôn trường view nếu có

            # 2. Khởi tạo mảng 14 nhãn mới toàn giá trị -100 (Thay cho 0)
            new_labels = [-100] * 14
            
            # 3. Ánh xạ từ nhãn NIH sang CheXpert
            has_positive = False
            for nih_idx, val in enumerate(labels[:14]):
                if val == 1 and nih_idx in NIH_TO_CHEX_MAP:
                    chex_idx = NIH_TO_CHEX_MAP[nih_idx]
                    new_labels[chex_idx] = 1
                    has_positive = True
            
            # Nếu tất cả đều là -100, có thể coi là No Finding (Index 0)
            if not has_positive:
                new_labels[8] = 1

            row["labels"] = new_labels
            row["report"] = labels_to_report(new_labels)
            # --- KẾT THÚC KHÚC THAY THẾ ---
            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")
            processed += 1

    return processed, skipped


def main() -> int:
    args = parse_args()
    processed, skipped = process_jsonl(args.input_jsonl.resolve(), args.output_jsonl.resolve())
    print(f"Processed rows: {processed}")
    print(f"Skipped rows: {skipped}")
    print(f"Output: {args.output_jsonl.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
