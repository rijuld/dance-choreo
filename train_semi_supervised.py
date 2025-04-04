import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import numpy as np
import os
import json
import time
from pathlib import Path

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

def main(verbose=1):
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
    
    # Load all data splits (train, validation, test)
    if config_time["train_ratio"] and config_time["train_lab_frac"]:
        # Time effort data
        (
            labelled_data_train_time,
            labels_train_time,
            unlabelled_data_train_time,
            labelled_data_valid_time,
            labels_valid_time,
            labelled_data_test_time,
            labels_test_time,
            unlabelled_data_test_time
        ) = datasets.get_model_data(config_time)
        
        # Space effort data
        (
            labelled_data_train_space,
            labels_train_space,
            unlabelled_data_train_space,
            labelled_data_valid_space,
            labels_valid_space,
            labelled_data_test_space,
            labels_test_space,
            unlabelled_data_test_space
        ) = datasets.get_model_data(config_space)
    else:
        # Time effort data
        (
            labelled_data_train_time,
            labels_train_time,
            unlabelled_data_train_time,
            labelled_data_valid_time,
            labels_valid_time,
            labelled_data_test_time,
            labels_test_time,
            unlabelled_data_test_time
        ) = datasets.get_model_specific_data(config_time)
        
        # Space effort data
        (
            labelled_data_train_space,
            labels_train_space,
            unlabelled_data_train_space,
            labelled_data_valid_space,
            labels_valid_space,
            labelled_data_test_space,
            labels_test_space,
            unlabelled_data_test_space
        ) = datasets.get_model_specific_data(config_space)
    
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
    
    # Create mask for labeled data (to exclude from unlabeled set) - Optimized version
    print("Filtering unlabeled sequences...")
    start_time = time.time()
    
    # Extract first frame of each sequence for faster comparison
    all_first_frames = all_sequences[:, 0, :]
    labeled_first_frames = np.array([seq[0] for seq in pose_sequences])
    
    # Use vectorized operations for matching instead of nested loops
    labeled_indices = set()
    
    # Process in batches to avoid memory issues with large datasets
    batch_size = 1000
    total_sequences = len(all_sequences)
    
    for batch_start in range(0, total_sequences, batch_size):
        batch_end = min(batch_start + batch_size, total_sequences)
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_sequences + batch_size - 1)//batch_size}")
        
        # Get current batch of first frames
        batch_first_frames = all_first_frames[batch_start:batch_end]
        
        # Compute distances between all pairs in current batch and labeled sequences
        # This is more efficient than calling np.allclose in nested loops
        for i, frame in enumerate(batch_first_frames):
            # Compute L2 norm (Euclidean distance) between this frame and all labeled frames
            distances = np.linalg.norm(labeled_first_frames - frame, axis=1)
            # If any distance is below threshold, consider it a match
            if np.any(distances < 1e-4):  # Equivalent to allclose with atol=1e-5
                labeled_indices.add(batch_start + i)
    
    # Filter out labeled sequences to get unlabeled ones
    unlabeled_indices = np.array([i for i in range(len(all_sequences)) if i not in labeled_indices])
    print(f"Found {len(unlabeled_indices)} unlabeled sequences out of {len(all_sequences)} total sequences")
    
    # Use direct indexing which is faster than creating a new array
    unlabeled_sequences = all_sequences[unlabeled_indices]
    unlabeled_tensor = torch.tensor(unlabeled_sequences, dtype=torch.float32)
    
    print(f"Filtering completed in {time.time() - start_time:.2f} seconds")
    
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
    num_classes = 9  # 3x3 matrix of time and space combinations
    classifier = Classifier(input_dim=256, num_classes=num_classes).to(device)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets and dataloaders
    print("\nPreparing dataloaders...")
    
    # Create labeled dataset for contrastive learning (pose-text pairs)
    # For contrastive learning, we use indices as labels since we're just matching text with poses
    contrastive_dataset = TensorDataset(pose_tensor, torch.tensor(np.arange(len(combined_text_labels))))
    labeled_loader = DataLoader(
        contrastive_dataset, 
        batch_size=16, 
        shuffle=True
    )
    
    # Create a separate dataset for classifier training with actual class labels
    # Extract time effort labels (0, 1, 2) for classifier training
    classifier_labels = []
    for (pose_batch_time, pose_batch_space), (label_batch_time, label_batch_space) in zip(
            zip(labelled_data_train_time, labelled_data_train_space),
            zip(labels_train_time, labels_train_space)):
        
        # Process each item in the batch
        for i in range(len(label_batch_time)):
            # Get both time and space labels (1-indexed)
            time_label_idx = int(label_batch_time[i].item()) + 1
            space_label_idx = int(label_batch_space[i].item()) + 1
            
            # Convert to 0-indexed classes for both dimensions
            # Time: Sustained (0), Neutral (1), Quick (2)
            if time_label_idx == 1:      # Sustained
                time_class = 0
            elif time_label_idx == 2:    # Neutral
                time_class = 1
            elif time_label_idx == 3:    # Quick
                time_class = 2
            else:                        # Unknown
                time_class = 1           # Default to neutral
            
            # Space: Indirect (0), Neutral (1), Direct (2)
            if space_label_idx == 1:     # Indirect
                space_class = 0
            elif space_label_idx == 2:   # Neutral
                space_class = 1
            elif space_label_idx == 3:   # Direct
                space_class = 2
            else:                        # Unknown
                space_class = 1          # Default to neutral
            
            # Combine classes into a single index (3x3 matrix)
            # Formula: time_class * 3 + space_class
            # This creates 9 distinct classes (0-8) representing all combinations
            combined_class = time_class * 3 + space_class
            
            classifier_labels.append(combined_class)
    
    # Create classifier dataset with proper class labels
    classifier_dataset = TensorDataset(pose_tensor, torch.tensor(classifier_labels))
    classifier_loader = DataLoader(
        classifier_dataset, 
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
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_path = checkpoint_dir / "semi_supervised_checkpoint.pt"
    start_epoch = 0
    training_history = []
    
    if checkpoint_path.exists():
        print(f"\nFound existing checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        pose_encoder.load_state_dict(checkpoint['pose_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        pose_projector.load_state_dict(checkpoint['pose_projector'])
        text_projector.load_state_dict(checkpoint['text_projector'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        training_history = checkpoint.get('history', [])
        print(f"Resuming training from epoch {start_epoch}")
    
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
        epochs=0,
        device=device,
        verbose=2,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        training_history=training_history,
        text_labels=combined_text_labels
    )
    
    # Save the trained models
    print("\nSaving final models...")
    models_dir = checkpoint_dir / "final_models"
    models_dir.mkdir(exist_ok=True)
    
    torch.save(pose_encoder.state_dict(), models_dir / "pose_encoder_semi.pth")
    torch.save(text_encoder.state_dict(), models_dir / "text_encoder_semi.pth")
    torch.save(pose_projector.state_dict(), models_dir / "pose_projector.pth")
    torch.save(text_projector.state_dict(), models_dir / "text_projector.pth")
    
    # Also save a copy in the root directory for backward compatibility
    torch.save(pose_encoder.state_dict(), "pose_encoder_semi.pth")
    torch.save(text_encoder.state_dict(), "text_encoder_semi.pth")
    torch.save(pose_projector.state_dict(), "pose_projector.pth")
    torch.save(text_projector.state_dict(), "text_projector.pth")
    
    # Optional: Train a classifier on top of the encoder
    # Uncomment and modify as needed
    
    print("\nTraining classifier on top of encoder...")
    # Create classifier and dataset for classification
    classifier = Classifier(input_dim=256, num_classes=9).to(device)
    
    # Create dataloaders for classification
    # Create validation dataset from validation data
    val_pose_sequences = []
    val_labels = []
    val_text_labels = []
    
    # Process validation data similar to how we processed training data
    for (pose_batch_time, pose_batch_space), (label_batch_time, label_batch_space) in zip(
            zip(labelled_data_valid_time, labelled_data_valid_space),
            zip(labels_valid_time, labels_valid_space)):
        
        # Process each item in the batch
        for i in range(len(label_batch_time)):
            # Store corresponding pose sequence
            val_pose_sequences.append(pose_batch_time[i].numpy())
            
            # Store label (using combined time and space efforts)
            # Convert to 0-indexed for classifier (0-8) using both time and space indices
            time_label_idx = int(label_batch_time[i].item()) + 1  # Convert to 1-indexed
            space_label_idx = int(label_batch_space[i].item()) + 1  # Convert to 1-indexed
            
            # Create text label (same format as training data)
            label_str = f"{time_effort_map[time_label_idx]} and {space_effort_map[space_label_idx]}"
            val_text_labels.append(label_str)
            
            # Map to 0-indexed classes for both dimensions
            # Time: Sustained (0), Neutral (1), Quick (2)
            if time_label_idx == 1:      # Sustained
                time_class = 0
            elif time_label_idx == 2:    # Neutral
                time_class = 1
            elif time_label_idx == 3:    # Quick
                time_class = 2
            else:                        # Unknown
                time_class = 1           # Default to neutral
            
            # Space: Indirect (0), Neutral (1), Direct (2)
            if space_label_idx == 1:     # Indirect
                space_class = 0
            elif space_label_idx == 2:   # Neutral
                space_class = 1
            elif space_label_idx == 3:    # Direct
                space_class = 2
            else:                        # Unknown
                space_class = 1          # Default to neutral
            
            # Combine classes into a single index (3x3 matrix)
            # Formula: time_class * 3 + space_class
            # This creates 9 distinct classes (0-8) representing all combinations
            combined_class = time_class * 3 + space_class
            
            val_labels.append(combined_class)
    
    # Convert validation pose sequences to tensor
    val_pose_sequences = np.array(val_pose_sequences)
    val_pose_tensor = torch.tensor(val_pose_sequences)
    val_labels_tensor = torch.tensor(val_labels)
    
    # Create validation dataset and loader
    val_dataset = TensorDataset(val_pose_tensor, val_labels_tensor)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False
    )
    
    print(f"Prepared {len(val_labels)} validation samples")
    
    # Train classifier with checkpoint support
    train_classifier(
        pose_encoder=pose_encoder,
        classifier=classifier,
        train_loader=classifier_loader,  # Use the classifier loader with proper class labels
        val_loader=val_loader,
        epochs=40,
        device=device,
        verbose=2,
        checkpoint_dir=checkpoint_dir,  # Pass checkpoint directory
        start_epoch=0,
        training_history=None
    )

    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()