import pandas as pd
import os

def create_split_csv(train_path, val_path, test_path, output_name="output_split.csv"):
    """
    Tạo file csv tổng hợp patient_id và phân loại split từ 3 file train, val, test.
    """
    
    # Danh sách cấu hình đầu vào: (đường dẫn, nhãn split)
    configs = [
        (train_path, 'train'),
        (val_path, 'val'),
        (test_path, 'test')
    ]
    
    list_df = []
    
    for path, split_label in configs:
        if os.path.exists(path):
            print(f"--- Đang xử lý file: {path} ---")
            # Chỉ load cột subject_id để tiết kiệm bộ nhớ
            df_temp = pd.read_csv(path, usecols=['subject_id'])
            
            # Loại bỏ trùng lặp trong nội bộ file đó
            df_unique = df_temp[['subject_id']].drop_duplicates()
            
            # Đổi tên cột thành patient_id và gán nhãn split
            df_unique = df_unique.rename(columns={'subject_id': 'patient_id'})
            df_unique['split'] = split_label
            
            list_df.append(df_unique)
        else:
            print(f"Cảnh báo: Không tìm thấy file {path}")

    if not list_df:
        print("Không có dữ liệu để xử lý.")
        return

    # Gộp 3 phần lại thành một DataFrame tổng
    df_final = pd.concat(list_df, ignore_index=True)
    
    # Sắp xếp theo patient_id tăng dần
    df_final = df_final.sort_values(by='patient_id').reset_index(drop=True)
    
    # Lưu file
    df_final.to_csv(output_name, index=False)
    print(f"\nĐã tạo thành công file: {output_name}")
    print(f"Tổng số bệnh nhân duy nhất: {len(df_final)}")
    print(df_final['split'].value_counts()) # Log số lượng mỗi loại

# --- Cấu hình đường dẫn file của bạn ---
# Bạn thay đổi tên file tương ứng ở đây
train_csv = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\splits\\train.csv'
val_csv = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\splits\\valid.csv'
test_csv = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\splits\\test.csv'

create_split_csv(train_csv, val_csv, test_csv)