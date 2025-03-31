import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import numpy as np

from utils import datasets
from utils.config import default_config
from models.model import TextEncoder, PoseEncoder, DEVICE
from models.semi_supervised import (
    ProjectorHead, 
    Classifier, 
    SemiSupervisedDataset,
    train_semi_supervised,
    train_classifier
)

# Define mappings (same as in main.py)
time_effort_map = {
    1: "Sustained",
    2: "Neutral Time",
    3: "Sudden/Quick",
    4: "Unknown Time"
}

space_effort_map = {
    1: "Indirect",
    2: "Neutral Space",
    3: "Direct",
    4: "Unknown Space"
}

# Helper: Create config per effort (same as in main.py)
def get_config(effort):
    cfg = {k: getattr(default_config, k) for k in dir(default_config) if not k.startswith("__") and not callable(getattr(default_config, k))}
    cfg["effort"] = effort
    return cfg

def main():
    print("\n" + "=" * 50)
    print("Starting Semi-Supervised Training")
    print("=" * 50 + "\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    config_time = get_config("time")
    config_space = get_config("space")
    
    # Load labeled data
    if config_time["train_ratio"] and config_time["train_lab_frac"]:
        labelled_data_train_time, labels_train_time, *_ = datasets.get_model_data(config_time)
        labelled_data_train_space, labels_train_space, *_ = datasets.get_model_data(config_space)
    else:
        labelled_data_train_time, labels_train_time, *_ = datasets.get_model_specific_data(config_time)
        labelled_data_train_space, labels_train_space, *_ = datasets.get_model_specific_data(config_space)
    
    # Combine labels
    combined_text_labels = []
    pose_sequences = []
    
    # Process each batch
    for (pose_batch_time, pose_batch_space), (label_batch_time, label_batch_space) in zip(
            zip(labelled_data_train_time, labelled_data_train_space),
            zip(labels_train_time, labels_train_space)):
        
        # Process each item in the batch
        for i in range(len(label_batch_time)):
            # Add 1 to the label indices since they are 0-indexed in the dataset but 1-indexed in our mapping
            time_label_idx = int(label_batch_time[i].item()) + 1
            space_label_idx = int(label_batch_space[i].item()) + 1
            
            # Create text label
            label_str = f"{time_effort_map[time_label_idx]} and {space_effort_map[space_label_idx]}"
            combined_text_labels.append(label_str)
            
            # Store corresponding pose sequence
            pose_sequences.append(pose_batch_time[i].numpy())
    
    # Convert pose sequences to tensor
    pose_sequences = np.array(pose_sequences)
    pose_tensor = torch.tensor(pose_sequences)
    
    print(f"Prepared {len(combined_text_labels)} labeled text-pose pairs")
    
    # Load unlabeled data (using all available data and removing labeled ones)
    print("\nLoading unlabeled data...")
    # Load raw data
    all_data, all_data_centered = datasets.load_raw()
    
    # Create sequences from all data
    seq_len = default_config.seq_len
    
    # Reshape the data to match the expected format for sequify_all_data
    # The function expects [n_poses, input_dim] but we have [n_poses, n_joints, 3]
    n_poses, n_joints, dims = all_data_centered.shape
    all_data_reshaped = all_data_centered.reshape(n_poses, n_joints * dims)
    
    # Now create sequences
    all_sequences = datasets.sequify_all_data(all_data_reshaped, seq_len, augmentation_factor=1)
    
    # Convert to tensor
    all_sequences_tensor = torch.tensor(all_sequences, dtype=torch.float32)
    
    # Create mask for labeled data (to exclude from unlabeled set)
    labeled_indices = set()
    for i, seq in enumerate(all_sequences):
        for labeled_seq in pose_sequences:
            # Check if sequences match (simple check - can be improved)
            if np.allclose(seq[0], labeled_seq[0], atol=1e-5):
                labeled_indices.add(i)
                break
    
    # Filter out labeled sequences to get unlabeled ones
    unlabeled_indices = [i for i in range(len(all_sequences)) if i not in labeled_indices]
    unlabeled_sequences = all_sequences[unlabeled_indices]
    unlabeled_tensor = torch.tensor(unlabeled_sequences, dtype=torch.float32)
    
    print(f"Prepared {len(unlabeled_sequences)} unlabeled pose sequences")
    
    # Initialize model components
    print("\nInitializing models...")
    
    # Encoders
    pose_encoder = PoseEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    
    # Projector heads for contrastive learning
    pose_projector = ProjectorHead(input_dim=256, hidden_dim=512, output_dim=128).to(device)
    text_projector = ProjectorHead(input_dim=256, hidden_dim=512, output_dim=128).to(device)
    
    # Classifier for evaluation
    num_classes = 3  # Assuming 3 classes for each effort (excluding Unknown)
    classifier = Classifier(input_dim=256, num_classes=num_classes).to(device)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets and dataloaders
    print("\nPreparing dataloaders...")
    
    # Labeled dataset (pose-text pairs)
    labeled_dataset = TensorDataset(pose_tensor, torch.tensor(np.arange(len(combined_text_labels))))
    labeled_loader = DataLoader(
        labeled_dataset, 
        batch_size=16, 
        shuffle=True
    )
    
    # Unlabeled dataset (poses only)
    unlabeled_loader = DataLoader(
        unlabeled_tensor,
        batch_size=16,
        shuffle=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(pose_encoder.parameters()) + 
        list(text_encoder.parameters()) + 
        list(pose_projector.parameters()) + 
        list(text_projector.parameters()),
        lr=1e-4
    )
    
    # Train with semi-supervised learning
    print("\nStarting semi-supervised training...")
    train_semi_supervised(
        pose_encoder=pose_encoder,
        text_encoder=text_encoder,
        pose_projector=pose_projector,
        text_projector=text_projector,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epochs=10,
        device=device,
        verbose=2
    )
    
    # Save the trained models
    print("\nSaving models...")
    torch.save(pose_encoder.state_dict(), "pose_encoder_semi.pth")
    torch.save(text_encoder.state_dict(), "text_encoder_semi.pth")
    torch.save(pose_projector.state_dict(), "pose_projector.pth")
    torch.save(text_projector.state_dict(), "text_projector.pth")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()