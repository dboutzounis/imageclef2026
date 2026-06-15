import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer

class ImageCLEFmedDataset(Dataset):
    """
    Custom PyTorch Dataset for the Concept Detection Task.
    Decoupled from file paths; initialized directly with a DataFrame.
    """
    def __init__(self, df, image_dir, mlb, transform=None, cui_mapping=None):
        self.data_df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mlb = mlb
        self.cui_mapping = cui_mapping or {}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_df.iloc[idx]
        img_id = row['ID']

        # Parse semicolon-separated CUIs
        cui_string = row['CUIs']
        cuis = str(cui_string).split(';') if pd.notna(cui_string) else []

        # Map CUIs to Canonical names (e.g., C0920367 -> Tomography, Optical Coherence)
        concept_names = [self.cui_mapping.get(cui, cui) for cui in cuis]

        # Load image
        img_name = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Binarize labels
        if self.mlb is not None:
            binary_labels = self.mlb.transform([cuis])[0]
            label_tensor = torch.tensor(binary_labels, dtype=torch.float32)
        else:
            label_tensor = torch.empty(0)  

        return {
            'id': img_id,
            'image': image,
            'labels': label_tensor,
            'cuis': ";".join(cuis),
            'concept_names': ";".join(concept_names)
        }

def get_transforms(img_size=224):
    """Returns standard validation/test transforms."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(concepts_csv, test_csv, image_dir, cui_mapping_csv=None, batch_size=16, num_workers=4, img_size=224):
    """
    Reads CSVs, handles train/val/test splitting, and returns DataLoaders.
    Accepts explicit file paths to be easily driven by command-line arguments.
    """
    # Load DataFrames
    all_concepts_df = pd.read_csv(concepts_csv)
    test_df = pd.read_csv(test_csv)
    
    cui_mapping = {}
    if cui_mapping_csv and os.path.exists(cui_mapping_csv):
        cui_map_df = pd.read_csv(cui_mapping_csv)
        cui_mapping = dict(zip(cui_map_df['CUI'], cui_map_df['Canonical name']))

    test_ids = set(test_df['ID'].tolist())
    print(f"Test samples loaded: {len(test_ids)}")

    # Isolate training and validation sets (preventing test leakage)
    train_df = all_concepts_df[
        all_concepts_df['ID'].str.contains('train', case=False, na=False) &
        ~all_concepts_df['ID'].isin(test_ids)
    ]
    
    valid_df = all_concepts_df[
        all_concepts_df['ID'].str.contains('valid', case=False, na=False)
    ]

    # Fit MultiLabelBinarizer strictly on the training CUIs
    all_train_cuis = train_df['CUIs'].dropna().apply(lambda x: str(x).split(';')).tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(all_train_cuis)

    transform = get_transforms(img_size=img_size)

    # Initialize Datasets
    train_dataset = ImageCLEFmedDataset(train_df, image_dir, mlb, transform, cui_mapping)
    val_dataset   = ImageCLEFmedDataset(valid_df, image_dir, mlb, transform, cui_mapping)
    test_dataset  = ImageCLEFmedDataset(test_df,  image_dir, mlb, transform, cui_mapping)

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Valid Dataset: {len(val_dataset)} samples")

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, mlb