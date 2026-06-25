import os
import json
import pandas as pd
import numpy as np

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

def load_jsonl(path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def analyze_dataset(name, df):
    """Perform EDA on a single dataset (filtered to train split)."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR {name.upper()}")
    print(f"{'='*80}")
    
    # Filter to train split only
    df_train = df[df['split'] == 'train'].copy()
    print(f"\nTotal train samples: {len(df_train)}")
    
    # ========== 1. PREVALENCE ANALYSIS ==========
    print(f"\n{'-'*80}")
    print("1. DISEASE PREVALENCE (Tỷ lệ mắc bệnh)")
    print(f"{'-'*80}")
    
    prevalence = []
    for idx, label_name in enumerate(CHEXPERT_LABELS):
        positive_count = 0
        for labels in df_train['labels'].dropna():
            if isinstance(labels, list) and len(labels) > idx and labels[idx] == 1:
                positive_count += 1
        percentage = (positive_count / len(df_train)) * 100 if len(df_train) > 0 else 0
        prevalence.append({
            'Label': label_name,
            'Positive Count': positive_count,
            'Prevalence %': round(percentage, 2)
        })
    
    prevalence_df = pd.DataFrame(prevalence).sort_values('Prevalence %', ascending=False)
    print(prevalence_df.to_string(index=False))
    
    rare_diseases = prevalence_df[prevalence_df['Prevalence %'] < 5]
    print(f"\nRare diseases (< 5% prevalence):")
    if len(rare_diseases) > 0:
        print(rare_diseases[['Label', 'Prevalence %']].to_string(index=False))
    else:
        print("  None")
    
    # ========== 2. REPORT WORD COUNT ANALYSIS ==========
    print(f"\n{'-'*80}")
    print("2. REPORT WORD COUNT STATISTICS (Độ dài Report)")
    print(f"{'-'*80}")
    
    df_train['word_count'] = df_train['report'].fillna('').apply(lambda x: len(str(x).split()))
    
    word_count_stats = {
        'Min': int(df_train['word_count'].min()),
        'Max': int(df_train['word_count'].max()),
        'Mean': round(df_train['word_count'].mean(), 2),
        'Median (50%)': int(df_train['word_count'].median()),
        '25th Percentile': int(df_train['word_count'].quantile(0.25)),
        '75th Percentile': int(df_train['word_count'].quantile(0.75)),
    }
    
    for stat_name, stat_value in word_count_stats.items():
        print(f"  {stat_name:20s}: {stat_value}")
    
    # ========== 3. IMAGES/STUDIES PER PATIENT DISTRIBUTION ==========
    print(f"\n{'-'*80}")
    print("3. STUDIES/IMAGES PER PATIENT DISTRIBUTION (Số lần chụp trên mỗi bệnh nhân)")
    print(f"{'-'*80}")
    
    studies_per_patient = df_train.groupby('patient_id').size()
    distribution = studies_per_patient.value_counts().sort_index()
    
    dist_data = []
    total_patients = distribution.sum()
    for num_studies, num_patients in distribution.items():
        dist_data.append({
            'Num Studies': num_studies,
            'Num Patients': num_patients,
            'Percentage': round((num_patients / total_patients) * 100, 2)
        })
    
    dist_df = pd.DataFrame(dist_data)
    print(dist_df.to_string(index=False))
    
    print(f"\n  Summary:")
    print(f"    Total unique patients: {len(studies_per_patient)}")
    print(f"    Patients with 1 study:   {(studies_per_patient == 1).sum()}")
    print(f"    Patients with 2 studies: {(studies_per_patient == 2).sum()}")
    print(f"    Patients with 3+ studies: {(studies_per_patient >= 3).sum()}")


def main():
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - METADATA FILTERING THRESHOLDS")
    print("="*80)
    
    # Load MIMIC data
    print("\nLoading MIMIC metadata...")
    mimic_data = load_jsonl(r"C:\Users\dhint\CHEX-DATA\MyChex\data\mimic_metadata_final.jsonl")
    df_mimic = pd.DataFrame(mimic_data)
    print(f"  Loaded {len(df_mimic)} records")
    
    # Load CheXplus data
    print("Loading CheXplus metadata...")
    chexplus_data = load_jsonl(r"C:\Users\dhint\CHEX-DATA\MyChex\data\chexplpus_metadata_final.jsonl")
    df_chexplus = pd.DataFrame(chexplus_data)
    print(f"  Loaded {len(df_chexplus)} records")
    
    # Analyze MIMIC
    analyze_dataset("MIMIC", df_mimic)
    
    # Analyze CheXplus
    analyze_dataset("CheXplus", df_chexplus)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
