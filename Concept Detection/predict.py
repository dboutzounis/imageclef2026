import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from huggingface_hub import hf_hub_download

from dataset import ImageCLEFmedDataset, get_transforms
from models import ConceptDetectionModel, ConceptCountRegressor

def load_weights(repo_id, filename, source='auto', weights_dir='./models', hf_token=None):
    local_path = os.path.join(weights_dir, os.path.basename(filename))
    
    if source in ['auto', 'local']:
        if os.path.exists(local_path):
            print(f"-> Loading local weight file from: {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
            return checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    if source in ['auto', 'hf']:
        print(f"-> Fetching {filename} from Hugging Face Hub repo: {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

def fetch_mlb_classes(repo_id, local_path, source='auto', hf_token=None):
    filename = os.path.basename(local_path)
    if source in ['auto', 'local'] and os.path.exists(local_path):
        print(f"-> Loading local MLB array from: {local_path}")
        return np.load(local_path, allow_pickle=True)
        
    if source in ['auto', 'hf']:
        print(f"-> Fetching {filename} from Hugging Face Hub...")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        return np.load(downloaded_path, allow_pickle=True)

def fetch_thresholds(repo_id, local_path, source='local', hf_token=None):
    filename = os.path.basename(local_path)
    if source in ['auto', 'local'] and os.path.exists(local_path):
        print(f"-> Loading local thresholds from: {local_path}")
        with open(local_path, "r") as f:
            return json.load(f)
            
    if source in ['auto', 'hf']:
        print(f"-> Fetching {filename} from Hugging Face Hub...")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        with open(downloaded_path, "r") as f:
            return json.load(f)
            
    raise ValueError(f"Could not resolve thresholds file using source '{source}'")

def apply_conformal_rescue(binary_preds, ensemble_probs, predicted_count, q_val):
    preds = binary_preds.copy() 
    current_count = preds.sum()
    lower_bound = max(0, int(round(predicted_count - q_val)))
    upper_bound = int(round(predicted_count + q_val))
    
    if current_count < lower_bound:
        missing_count = lower_bound - current_count
        unpredicted_indices = np.where(preds == 0)[0]
        sorted_unpredicted = unpredicted_indices[np.argsort(ensemble_probs[unpredicted_indices])[::-1]]
        for idx in sorted_unpredicted[:missing_count]:
            preds[idx] = 1
            
    elif current_count > upper_bound:
        surplus_count = current_count - upper_bound
        predicted_indices = np.where(preds == 1)[0]
        sorted_predicted = predicted_indices[np.argsort(ensemble_probs[predicted_indices])]
        for idx in sorted_predicted[:surplus_count]:
            preds[idx] = 0
            
    return preds

@torch.no_grad()
def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if len(args.backbones) == 1 and args.strategy != 'single':
        print("\n[Info] Only one model provided. Auto-switching strategy to 'single'.")
        args.strategy = 'single'
    elif args.strategy == 'single' and len(args.backbones) > 1:
        raise ValueError("The 'single' strategy requires exactly one backbone and weight file. You provided multiple.")

    if args.conformal == 'before' and args.strategy == 'soft_voting':
        print("\n[Notice] Soft Voting averages continuous probabilities. Conformal Rescue modifies discrete predictions. Thus, applying it 'before' soft voting has no mathematical effect. Auto-switching '--conformal' to 'after'.")
        args.conformal = 'after'

    print(f"Running inference on: {device} | Strategy: {args.strategy}")
    if args.conformal != 'none':
        print(f"Conformal Rescue Placement: {args.conformal.upper()} ensembling")

    if len(args.backbones) != len(args.weights):
        raise ValueError("Backbones must match weights provided.")

    hf_token = os.environ.get("HF_TOKEN")
    
    # Load Thresholds using the new dynamic fetcher
    threshold_config = fetch_thresholds(args.repo_id, args.thresholds_path, args.thresholds_source, hf_token)
    q_val = threshold_config.get("uncertainty_calibration", {}).get("conformal_q_val", 0.0)
    
    mlb_classes = fetch_mlb_classes(args.repo_id, args.mlb_classes_path, args.weights_source, hf_token)
    num_classes = len(mlb_classes)
    
    test_df = pd.read_csv(args.test_csv)
    transform = get_transforms(img_size=224)
    test_dataset = ImageCLEFmedDataset(test_df, args.image_dir, mlb=None, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    models_to_ensemble = []
    
    for backbone, weight_file in zip(args.backbones, args.weights):
        model = ConceptDetectionModel(backbone_name=backbone, num_classes=num_classes, pool_type=args.pooling)
        state_dict = load_weights(args.repo_id, weight_file, args.weights_source, args.weights_dir, hf_token)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        # Robust Underscore-Agnostic Threshold Lookup
        weight_key = os.path.basename(weight_file).replace('.pth', '')
        weight_key_clean = weight_key.replace('_', '').lower()
        ind_thresh = None
        
        for config_category in ["cross_seed_models", "mccv_splits"]:
            for json_key, json_val in threshold_config.get(config_category, {}).items():
                if json_key.replace('_', '').lower() == weight_key_clean:
                    ind_thresh = json_val
                    break
            if ind_thresh is not None:
                break
                
        if ind_thresh is None:
            ind_thresh = args.ensemble_threshold
            print(f"[Warning] Exact threshold not found for {weight_key}. Falling back to default: {ind_thresh}")
            
        print(f"Loaded model '{backbone}' with threshold: {ind_thresh:.4f}")
        models_to_ensemble.append({'model': model, 'threshold': ind_thresh})
        
    if args.conformal != 'none':
        print("Initializing Conformal Regressor...")
        regressor = ConceptCountRegressor(backbone_name='efficientnet_v2_m') 
        reg_weights = load_weights(args.repo_id, "count_regressor/count_regressor_effnetv2_m.pth", args.weights_source, args.weights_dir, hf_token)
        regressor.load_state_dict(reg_weights, strict=False)
        regressor.to(device).eval()

    submission_data = []
    all_final_preds = []
    
    pbar = tqdm(test_loader, desc="Generating Predictions")
    for batch in pbar:
        images = batch['image'].to(device)
        img_ids = batch['id']
        
        batch_model_probs = []
        batch_model_preds = []
        
        for item in models_to_ensemble:
            logits = item['model'](images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= item['threshold']).astype(int)
            
            batch_model_probs.append(probs)
            batch_model_preds.append(preds)
            
        batch_model_probs = np.array(batch_model_probs) 
        batch_model_preds = np.array(batch_model_preds) 
        
        avg_probs = np.mean(batch_model_probs, axis=0)
        
        if args.conformal != 'none':
            count_logits = regressor(images)
            predicted_counts = count_logits.view(-1).cpu().numpy()
        
        for i, img_id in enumerate(img_ids):
            
            if args.strategy == 'single':
                base_preds = batch_model_preds[0, i]
                if args.conformal != 'none':
                    final_binary_preds = apply_conformal_rescue(base_preds, batch_model_probs[0, i], predicted_counts[i], q_val)
                else:
                    final_binary_preds = base_preds
                    
            elif args.strategy == 'soft_voting':
                base_preds = (avg_probs[i] >= args.ensemble_threshold).astype(int)
                if args.conformal != 'none': 
                    final_binary_preds = apply_conformal_rescue(base_preds, avg_probs[i], predicted_counts[i], q_val)
                else:
                    final_binary_preds = base_preds
                    
            elif args.strategy == 'dual_threshold':
                if args.conformal == 'before':
                    rescued_model_preds = []
                    for m in range(len(models_to_ensemble)):
                        m_preds = batch_model_preds[m, i]
                        m_probs = batch_model_probs[m, i]
                        rescued = apply_conformal_rescue(m_preds, m_probs, predicted_counts[i], q_val)
                        rescued_model_preds.append(rescued)
                    
                    votes = np.sum(rescued_model_preds, axis=0)
                    final_binary_preds = (votes >= args.dual_L).astype(int)
                    
                else:
                    votes = np.sum(batch_model_preds[:, i, :], axis=0)
                    base_preds = (votes >= args.dual_L).astype(int)
                    
                    if args.conformal == 'after':
                        final_binary_preds = apply_conformal_rescue(base_preds, avg_probs[i], predicted_counts[i], q_val)
                    else:
                        final_binary_preds = base_preds
            
            all_final_preds.append(final_binary_preds)
            pred_indices = np.where(final_binary_preds == 1)[0]
            predicted_cuis = [mlb_classes[idx] for idx in pred_indices]
            submission_data.append({'ID': img_id, 'CUIs': ";".join(list(dict.fromkeys(predicted_cuis)))})

    if args.evaluate:
        if 'CUIs' not in test_df.columns:
            print("\n[Warning] --evaluate flag used, but 'CUIs' column not found in CSV.")
        else:
            print("\nCalculating Evaluation Metrics...")
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer(classes=mlb_classes)
            mlb.fit([mlb_classes]) 
            gt_cuis_list = test_df['CUIs'].apply(lambda x: str(x).split(';') if pd.notna(x) else []).tolist()
            f1 = f1_score(mlb.transform(gt_cuis_list), np.array(all_final_preds), average='samples', zero_division=0)
            print(f"=====================================\nFinal Samples-Average F1: {f1:.4f}\n=====================================")

    print("\nFormatting submission file...")
    submission_df = pd.DataFrame(submission_data).drop_duplicates(subset=['ID'], keep='first')
    submission_df['temp_num'] = submission_df['ID'].str.extract(r'(\d+)$').astype(float)
    csv_filename = os.path.join(args.output_dir, "submission.csv")
    submission_df.sort_values(by='temp_num').drop(columns=['temp_num']).to_csv(csv_filename, index=False, encoding='utf-8', lineterminator='\n')
    print(f"Inference complete. File saved to {csv_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    
    # Threshold sourcing arguments
    parser.add_argument('--thresholds_path', type=str, default='thresholds.json', help="Filename or local path for thresholds config")
    parser.add_argument('--thresholds_source', type=str, choices=['local', 'hf', 'auto'], default='local', help="Where to fetch the thresholds JSON from")
    
    parser.add_argument('--mlb_classes_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='.')
    
    parser.add_argument('--weights_source', type=str, choices=['auto', 'local', 'hf'], default='auto')
    parser.add_argument('--weights_dir', type=str, default='./models')
    parser.add_argument('--repo_id', type=str, default='dboutzounis/imageclefmed-2026-concept-detection')
    
    parser.add_argument('--backbones', nargs='+', required=True)
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--pooling', type=str, default='gem', choices=['avg', 'max', 'gem'])
    
    parser.add_argument('--strategy', type=str, choices=['soft_voting', 'dual_threshold', 'single'], default='soft_voting')
    parser.add_argument('--ensemble_threshold', type=float, default=0.35)
    parser.add_argument('--dual_L', type=int, default=5)
    
    parser.add_argument('--conformal', type=str, choices=['none', 'before', 'after'], default='none')
    
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    run_inference(args)