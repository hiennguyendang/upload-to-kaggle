import json
import csv
from pathlib import Path

# Danh sách 14 nhãn chuẩn của CheXbert/CheXpert tương ứng với các index trong mảng labels
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", 
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", 
    "Pneumonia", "Pneumothorax", "Support Devices"
]

def generate_scene_graph(input_path: Path, output_path: Path):
    # Đảm bảo thư mục đầu ra tồn tại
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tiêu đề cột của file CSV
    headers = [
        "patient_id", "study_id", "study_key", "source_section", 
        "source_view", "anatomy", "observation", "presence", 
        "temporal_status", "bboxes"
    ]

    processed_rows = 0
    generated_records = 0

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:
        
        writer = csv.writer(f_out)
        writer.writerow(headers)  # Ghi tiêu đề

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            patient_id = data.get("patient_id", "")
            study_id = data.get("study_id", "")
            labels = data.get("labels", [])

            # Nếu thiếu thông tin cần thiết, bỏ qua dòng này
            if not patient_id or not study_id or not isinstance(labels, list):
                continue

            # Tạo các trường cố định
            study_key = f"{patient_id}_{study_id}"
            source_section = "label"
            source_view = "frontal"
            anatomy = ""
            temporal_status = ""
            bboxes = "[]" # Để [] cho đồng nhất với định dạng rỗng của các tập dữ liệu X-quang

            # Duyệt qua mảng labels, chỉ lấy những nhãn có giá trị 1 (present)
            for idx, val in enumerate(labels):
                if val == 1 and idx < len(CHEXPERT_LABELS):
                    observation = CHEXPERT_LABELS[idx].lower() # Viết thường tên bệnh cho đồng nhất
                    presence = "present"
                    
                    # Cấu trúc 1 dòng trong file csv
                    row = [
                        patient_id,
                        study_id,
                        study_key,
                        source_section,
                        source_view,
                        anatomy,
                        observation,
                        presence,
                        temporal_status,
                        bboxes
                    ]
                    writer.writerow(row)
                    generated_records += 1
                    
            processed_rows += 1

    print(f"Hoàn thành! Đã xử lý {processed_rows} ảnh.")
    print(f"Tổng số dòng Scene Graph được tạo ra: {generated_records}")
    print(f"File output được lưu tại: {output_path}")

if __name__ == "__main__":
    # Khai báo đường dẫn
    INPUT_JSONL = Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata.jsonl")
    OUTPUT_CSV = Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_scene_graph.csv")
    
    generate_scene_graph(INPUT_JSONL, OUTPUT_CSV)