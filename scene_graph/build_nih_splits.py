import pandas as pd
import random
from pathlib import Path

def process_nih_splits():
    # Đường dẫn file
    test_list_path = Path(r"C:\Users\dhint\CHEX-DATA\NIH\test_list.txt")
    train_val_list_path = Path(r"C:\Users\dhint\CHEX-DATA\NIH\train_val_list.txt")
    output_csv = Path(r"C:\Users\dhint\CHEX-DATA\NIH\patient_splits.csv")

    def get_unique_patients(file_path):
        if not file_path.exists():
            print(f"Cảnh báo: Không tìm thấy {file_path}")
            return set()
        
        patients = set()
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Lấy phần trước dấu gạch dưới: 00000001_000.png -> 00000001
                    p_id = line.split('_')[0]
                    patients.add(p_id)
        return patients

    # 1. Đọc dữ liệu
    test_patients = list(get_unique_patients(test_list_path))
    train_val_patients = list(get_unique_patients(train_val_list_path))

    # 2. Chia tập train_val thành train và val (tỉ lệ 7:1)
    # Tổng số phần là 7 + 1 = 8. Val chiếm 1/8.
    random.seed(42) # Đảm bảo kết quả giống nhau mỗi lần chạy
    random.shuffle(train_val_patients)
    
    split_idx = int(len(train_val_patients) * (9/10))
    train_patients = train_val_patients[:split_idx]
    val_patients = train_val_patients[split_idx:]

    # 3. Tạo danh sách dữ liệu để chuyển sang DataFrame
    data = []
    
    for p in train_patients:
        data.append({'patient_id': p, 'split': 'train'})
    
    for p in val_patients:
        data.append({'patient_id': p, 'split': 'val'})
        
    for p in test_patients:
        data.append({'patient_id': p, 'split': 'test'})

    # 4. Ghi ra CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"Đã xử lý xong!")
    print(f"- Train: {len(train_patients)} bệnh nhân")
    print(f"- Val: {len(val_patients)} bệnh nhân")
    print(f"- Test: {len(test_patients)} bệnh nhân")
    print(f"File lưu tại: {output_csv}")

if __name__ == "__main__":
    process_nih_splits()