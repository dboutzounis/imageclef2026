import os
import argparse
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import create_dataloaders
from models import ConceptDetectionModel

def set_seed(seed=42):
    """Ensures deterministic behavior across random initializations."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_probabilities(model, dataloader, device):
    """Extracts raw probabilities and labels for post-training threshold optimization."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Validation Probs"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device).float()
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    return np.vstack(all_labels), np.vstack(all_probs)

def find_best_global_threshold(y_true, y_prob, start=0.1, end=0.9, steps=81, device="cpu"):
    """
    Fast, tensor-based threshold sweep. 
    Exactly mirrors the Colab notebook's sample-averaged F1 calculation.
    """
    y_true_t = torch.tensor(y_true, dtype=torch.float32, device=device)
    y_prob_t = torch.tensor(y_prob, dtype=torch.float32, device=device)

    # Mask out completely empty samples for accurate sample-averaged metrics
    valid_mask = torch.sum(y_true_t, dim=1) > 0
    y_true_t = y_true_t[valid_mask]
    y_prob_t = y_prob_t[valid_mask]

    candidates = torch.linspace(start, end, steps, device=device)
    best_t = 0.5
    best_f1 = 0.0

    true_sum = torch.sum(y_true_t, dim=1)

    for t in tqdm(candidates, desc="Sweeping Global Thresholds"):
        y_pred_t = (y_prob_t >= t).float()

        tp = torch.sum(y_true_t * y_pred_t, dim=1)
        pred_sum = torch.sum(y_pred_t, dim=1)

        denominator = pred_sum + true_sum

        f1_per_sample = torch.where(
            denominator > 0,
            2.0 * tp / denominator,
            torch.zeros_like(tp)
        )

        current_f1 = torch.mean(f1_per_sample).item()

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_t = t.item()

    return best_t, best_f1

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device).float()
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Calculate fast 0.5 baseline predictions for epoch tracking
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(dataloader.dataset)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    train_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    return epoch_loss, train_f1

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device).float()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    val_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    return epoch_loss, val_f1

def main(args):
    set_seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, mlb = create_dataloaders(
        concepts_csv=args.concepts_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        cui_mapping_csv=args.cui_mapping_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(mlb.classes_)
    
    mlb_save_path = os.path.join(args.output_dir, "mlb_classes.npy")
    np.save(mlb_save_path, mlb.classes_)
    print(f"Saved MLB classes array to {mlb_save_path}")

    # 2. Initialize Model
    print(f"Initializing {args.backbone}...")
    model = ConceptDetectionModel(
        backbone_name=args.backbone,
        num_classes=num_classes, 
        pool_type=args.pooling
    )
    model = model.to(device)

    # 3. Setup Optimizer, Loss, and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    
    opt_choice = args.optimizer.lower()
    if opt_choice == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif opt_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif opt_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_choice}")
        
    print(f"Using {opt_choice.upper()} optimizer with LR={args.lr}")
    
    # Scheduler: Configurable LR reduction on plateau
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=args.lr_factor, 
        patience=args.lr_patience, 
    )

    # 4. Training Loop setup
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = os.path.join(args.output_dir, f"{args.backbone}_seed{args.seed}.pth")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Loss (Tr/Val): {train_loss:.4f}/{val_loss:.4f} | F1 @ 0.5 (Tr/Val): {train_f1:.4f}/{val_f1:.4f}")
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Early Stopping & Model Checkpointing based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f"New best Val Loss achieved ({best_val_loss:.4f})! Saving model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stopping_counter += 1
            print(f"No improvement in Val Loss. Early stopping patience: {early_stopping_counter}/{args.early_stopping_patience}")
            
        if early_stopping_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # 5. Post-Training Optimization
    training_time = time.time() - start_time
    print("-" * 40)
    print(f"Training Time: {training_time / 60:.2f} minutes")
    
    print(f"\nLoading best model for Threshold Optimization...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    y_true, y_probs = get_probabilities(model, val_loader, device)
    best_thresh, best_f1 = find_best_global_threshold(y_true, y_probs, device=device)
    
    print("=" * 40)
    print(f"FINAL OPTIMIZATION RESULTS ({args.backbone}_seed{args.seed})")
    print(f"Optimal Global Threshold : {best_thresh:.2f}")
    print(f"Peak Validation F1 Score : {best_f1:.6f}")
    print("=" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Concept Detection Model")
    parser.add_argument('--concepts_csv', type=str, required=True, help="Path to training concepts CSV")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to test concepts CSV (for exclusion)")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing .jpg/.png images")
    parser.add_argument('--cui_mapping_csv', type=str, default=None, help="Optional: Path to CUI mappings")
    parser.add_argument('--output_dir', type=str, default='./models', help="Where to save trained weights")
    
    parser.add_argument('--backbone', type=str, default='efficientnetb0', help="Model architecture")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for weight initialization")
    parser.add_argument('--pooling', type=str, default='gem', choices=['avg', 'max', 'gem'], help='Type of pooling layer to use')
    
    parser.add_argument('--epochs', type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'], help="Optimizer choice")
    
    parser.add_argument('--lr_factor', type=float, default=0.5, help="Factor to reduce learning rate by on plateau")
    parser.add_argument('--lr_patience', type=int, default=1, help="Patience for the LR reduction scheduler")
    
    parser.add_argument('--early_stopping_patience', type=int, default=3, help="Early stopping patience")
    parser.add_argument('--num_workers', type=int, default=2, help="Dataloader workers")
    
    args = parser.parse_args()
    main(args)