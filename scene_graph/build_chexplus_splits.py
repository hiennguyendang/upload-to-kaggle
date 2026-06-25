"""
Build CheXplus patient-level split scores from scene graph CSV.

Outputs to C:/Users/dhint/CHEX-DATA/CHEXPLUS/metadata:
- splits.csv
- split_score_value_counts.csv
- split_tier_value_counts.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    metadata_dir = Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata")
    scene_graph_csv = metadata_dir / "chexplus_scene_graph.csv"
    splits_csv = metadata_dir / "splits.csv"
    counts_csv = metadata_dir / "split_score_value_counts.csv"
    tier_counts_csv = metadata_dir / "split_tier_value_counts.csv"
    split_counts_csv = metadata_dir / "split_value_counts.csv"

    df = pd.read_csv(scene_graph_csv)

    for col in ["patient_id", "study_id", "anatomy", "presence", "temporal_status"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Spatial score per study:
    # count rows where anatomy is not null and presence in {present, absent}
    anatomy_ok = df["anatomy"].notna() & df["anatomy"].astype(str).str.strip().ne("")
    presence_ok = df["presence"].fillna("").astype(str).str.lower().isin(["present", "absent"])
    df["spatial_flag"] = (anatomy_ok & presence_ok).astype(int)

    # Temporal score per study:
    # count rows where temporal_status is not null/non-empty
    temporal_ok = df["temporal_status"].notna() & df["temporal_status"].astype(str).str.strip().ne("")
    df["temporal_flag"] = temporal_ok.astype(int)

    study_scores = (
        df.groupby(["patient_id", "study_id"], as_index=False)
        .agg(spatial_score=("spatial_flag", "sum"), temporal_score=("temporal_flag", "sum"))
    )

    patient_scores = (
        study_scores.groupby("patient_id", as_index=False)
        .agg(max_spatial_score=("spatial_score", "max"), max_temporal_score=("temporal_score", "max"))
        .sort_values("patient_id")
    )

    def classify_tier(row: pd.Series) -> str:
        if row["max_spatial_score"] >= 2 and row["max_temporal_score"] >= 4:
            return "gold"
        if row["max_temporal_score"] >= 1 and row["max_spatial_score"] >= 6:
            return "gold"
        return "silver"

    patient_scores["patient_tier"] = patient_scores.apply(classify_tier, axis=1)

    # --- ĐOẠN MỚI THAY THẾ ---
    import numpy as np

    # Bước 1: Tính toán số lượng cần thiết theo tỷ lệ 70/20/10 trên TỔNG số bệnh nhân
    total_patients = len(patient_scores)
    n_test = int(total_patients * 0.20)  # 20% cho Test
    n_val = int(total_patients * 0.10)   # 10% cho Val
    # Còn lại tự động là Train (70%)

    # Bước 2: Mặc định tất cả là 'train' (bao gồm cả Silver và Gold)
    patient_scores["split"] = "train"

    # Bước 3: Lấy danh sách index của những bệnh nhân nhóm 'gold'
    gold_indices = patient_scores[patient_scores["patient_tier"] == "gold"].index.tolist()

    # Kiểm tra nếu Gold không đủ để gánh Val và Test thì báo lỗi
    if len(gold_indices) < (n_test + n_val):
        raise ValueError(f"Nhóm Gold ({len(gold_indices)}) không đủ để chia cho Val+Test ({n_test + n_val})")

    # Bước 4: Bốc ngẫu nhiên từ nhóm Gold ra n_test người làm Test
    rs = np.random.RandomState(42)
    test_idx = rs.choice(gold_indices, size=n_test, replace=False)
    patient_scores.loc[test_idx, "split"] = "test"

    # Bước 5: Bốc tiếp từ những người Gold còn lại (trừ những người đã vào Test) ra n_val người làm Val
    remaining_gold = [i for i in gold_indices if i not in test_idx]
    val_idx = rs.choice(remaining_gold, size=n_val, replace=False)
    patient_scores.loc[val_idx, "split"] = "val"

    # Kết quả: Những người Gold còn dư và TOÀN BỘ nhóm Silver vẫn giữ nguyên là 'train'

    patient_scores = patient_scores[["patient_id", "split", "patient_tier", "max_spatial_score", "max_temporal_score"]]

    patient_scores.to_csv(splits_csv, index=False)

    spatial_counts = (
        patient_scores["max_spatial_score"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("score")
        .reset_index(name="count")
    )
    spatial_counts.insert(0, "metric", "max_spatial_score")

    temporal_counts = (
        patient_scores["max_temporal_score"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("score")
        .reset_index(name="count")
    )
    temporal_counts.insert(0, "metric", "max_temporal_score")

    pd.concat([spatial_counts, temporal_counts], ignore_index=True).to_csv(counts_csv, index=False)

    tier_counts = (
        patient_scores["patient_tier"]
        .value_counts(dropna=False)
        .rename_axis("patient_tier")
        .reset_index(name="count")
        .sort_values("patient_tier")
    )
    tier_counts.to_csv(tier_counts_csv, index=False)

    split_counts = (
        patient_scores["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
        .sort_values("split")
    )
    split_counts.to_csv(split_counts_csv, index=False)

    print(f"Wrote {splits_csv}")
    print(f"Wrote {counts_csv}")
    print(f"Wrote {tier_counts_csv}")
    print(f"Wrote {split_counts_csv}")
    print(f"Patients: {len(patient_scores)}")
    print("Tier counts:\n", tier_counts.to_string(index=False))
    print("Split counts:\n", split_counts.to_string(index=False))


if __name__ == "__main__":
    main()
