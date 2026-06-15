import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# The 4 most frequent concepts that mask rare pathology performance
DOMINANT_CONCEPTS = {'C0040405', 'C1306645', 'C0024485', 'C0041618'}

def filter_concepts(cui_string):
    """Parses the CUI string and removes the dominant high-frequency concepts."""
    if pd.isna(cui_string):
        return []
    cuis = str(cui_string).split(';')
    # Keep only the CUIs that are NOT in the dominant set
    return [cui for cui in cuis if cui not in DOMINANT_CONCEPTS]

def main(args):
    print("Loading ground truth and predictions...")
    
    # 1. Load Data
    gt_df = pd.read_csv(args.ground_truth_csv)
    pred_df = pd.read_csv(args.predictions_csv)
    
    # 2. Ensure both DataFrames are aligned by ID
    merged_df = pd.merge(gt_df, pred_df, on='ID', suffixes=('_gt', '_pred'))
    
    if len(merged_df) != len(pred_df):
        print(f"Warning: Only matched {len(merged_df)} out of {len(pred_df)} prediction rows to the ground truth.")
        
    # 3. Filter out the dominant concepts
    print("Filtering out top 4 high-frequency imaging modalities...")
    gt_filtered = merged_df['CUIs_gt'].apply(filter_concepts).tolist()
    pred_filtered = merged_df['CUIs_pred'].apply(filter_concepts).tolist()
    
    # 4. Binarize the remaining concepts
    # Fit the MLB on the combined set of all possible remaining concepts to ensure alignment
    mlb = MultiLabelBinarizer()
    mlb.fit(gt_filtered + pred_filtered)
    
    y_true = mlb.transform(gt_filtered)
    y_pred = mlb.transform(pred_filtered)
    
    # 5. Calculate the sample-averaged F1 Score
    long_tail_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    print("\n==========================================")
    print(f"Long-Tail F1 Score (Samples Avg): {long_tail_f1:.4f}")
    print("==========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate the Long-Tail F1 Score")
    parser.add_argument('--ground_truth_csv', type=str, required=True, help="Path to the original concepts.csv")
    parser.add_argument('--predictions_csv', type=str, required=True, help="Path to your generated submission.csv")
    
    args = parser.parse_args()
    main(args)