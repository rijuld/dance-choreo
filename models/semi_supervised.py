import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Constants
TEMPERATURE = 0.07

# Projector head for contrastive learning (similar to SimCLR)
class ProjectorHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

# Simple classifier for fine-tuning on labeled data
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

# Data augmentation functions for unlabeled data
def time_crop(sequence, min_crop_ratio=0.8):
    """Randomly crop a sequence in the time dimension"""
    seq_len = sequence.shape[0]
    crop_len = int(seq_len * min_crop_ratio)
    start_idx = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
    cropped = sequence[start_idx:start_idx + crop_len]
    # Resize back to original length using interpolation
    if isinstance(sequence, np.ndarray):
        return torch.nn.functional.interpolate(
            torch.from_numpy(cropped).unsqueeze(0).permute(0, 2, 1),
            size=seq_len,
            mode='linear'
        ).permute(0, 2, 1).squeeze(0).numpy()
    else:  # torch.Tensor
        return torch.nn.functional.interpolate(
            cropped.unsqueeze(0).permute(0, 2, 1),
            size=seq_len,
            mode='linear'
        ).permute(0, 2, 1).squeeze(0)

def jitter(sequence, scale=0.05):
    """Add random noise to the sequence"""
    if isinstance(sequence, np.ndarray):
        noise = np.random.normal(0, scale, sequence.shape)
        return sequence + noise
    else:  # torch.Tensor
        noise = torch.randn_like(sequence) * scale
        return sequence + noise

def rotation_augment(sequence, max_angle=15):
    """Apply small random rotation to the sequence"""
    # Convert degrees to radians
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    if isinstance(sequence, np.ndarray):
        seq_len, input_features = sequence.shape
        seq_reshaped = sequence.reshape((seq_len, -1, 3))
        rotated_seq = np.einsum("...j, ij->...i", seq_reshaped, rotation_mat)
        return rotated_seq.reshape((seq_len, input_features))
    else:  # torch.Tensor
        seq_len, input_features = sequence.shape
        seq_reshaped = sequence.reshape((seq_len, -1, 3))
        rotation_mat_tensor = torch.tensor(rotation_mat, dtype=sequence.dtype, device=sequence.device)
        rotated_seq = torch.einsum("...j, ij->...i", seq_reshaped, rotation_mat_tensor)
        return rotated_seq.reshape((seq_len, input_features))

# Apply multiple augmentations
def augment_sequence(sequence):
    """Apply a series of augmentations to create a positive pair"""
    # Apply random augmentations
    augmented = sequence
    
    # Randomly select which augmentations to apply
    if np.random.random() > 0.5:
        augmented = time_crop(augmented)
    if np.random.random() > 0.5:
        augmented = jitter(augmented)
    if np.random.random() > 0.5:
        augmented = rotation_augment(augmented)
        
    return augmented

# Dataset for semi-supervised learning
class SemiSupervisedDataset(Dataset):
    def __init__(self, labeled_poses, labels, unlabeled_poses=None):
        self.labeled_poses = labeled_poses
        self.labels = labels
        self.unlabeled_poses = unlabeled_poses
        self.has_unlabeled = unlabeled_poses is not None and len(unlabeled_poses) > 0
        
    def __len__(self):
        return len(self.labeled_poses) + (len(self.unlabeled_poses) if self.has_unlabeled else 0)
    
    def __getitem__(self, idx):
        if idx < len(self.labeled_poses):
            # Labeled data
            pose = self.labeled_poses[idx]
            label = self.labels[idx]
            return {
                'pose': pose,
                'label': label,
                'is_labeled': True
            }
        else:
            # Unlabeled data
            unlabeled_idx = idx - len(self.labeled_poses)
            pose = self.unlabeled_poses[unlabeled_idx]
            # Create augmented version for positive pair
            augmented_pose = augment_sequence(pose)
            return {
                'pose': pose,
                'augmented_pose': augmented_pose,
                'is_labeled': False
            }

# Contrastive loss for semi-supervised learning
def semi_supervised_contrastive_loss(z_i, z_j=None, labels=None, mask=None):
    """
    Compute contrastive loss for semi-supervised learning
    
    Args:
        z_i: Embeddings from encoder+projector
        z_j: Optional augmented embeddings (for unlabeled data)
        labels: Optional labels (for labeled data)
        mask: Boolean mask indicating which samples are labeled
    
    Returns:
        Loss value
    """
    batch_size = z_i.shape[0]
    device = z_i.device
    
    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    
    if z_j is not None:
        z_j = F.normalize(z_j, dim=1)
    
    # Initialize loss
    loss = 0.0
    
    # Process labeled data (supervised contrastive loss)
    if labels is not None and mask is not None and torch.any(mask):
        labeled_z = z_i[mask]
        labeled_labels = labels[mask]
        
        # Compute similarity matrix for labeled data
        sim_matrix = torch.matmul(labeled_z, labeled_z.T) / TEMPERATURE
        
        # Create mask for positive pairs (same label)
        label_mask = labeled_labels.unsqueeze(1) == labeled_labels.unsqueeze(0)
        # Remove self-similarity
        label_mask.fill_diagonal_(False)
        
        # For each anchor, compute loss over positive pairs
        n_labeled = labeled_z.shape[0]
        for i in range(n_labeled):
            if not torch.any(label_mask[i]):
                continue  # Skip if no positive pairs
                
            # Positive pairs (same label)
            pos_pairs = sim_matrix[i][label_mask[i]]
            
            # All pairs for denominator
            all_pairs = sim_matrix[i]
            
            # Compute log-softmax
            logits = torch.cat([pos_pairs, all_pairs])
            labels_i = torch.zeros(len(logits), device=device)
            labels_i[:len(pos_pairs)] = 1.0 / len(pos_pairs)
            
            # Compute loss
            loss_i = -torch.sum(labels_i * F.log_softmax(logits, dim=0))
            loss += loss_i
        
        # Normalize by number of labeled samples
        if n_labeled > 0:
            loss = loss / n_labeled
    
    # Process unlabeled data (SimCLR-style contrastive loss)
    if z_j is not None and mask is not None and torch.any(~mask):
        unlabeled_z_i = z_i[~mask]
        unlabeled_z_j = z_j[~mask]
        
        # Concatenate embeddings from both augmentations
        unlabeled_z = torch.cat([unlabeled_z_i, unlabeled_z_j], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(unlabeled_z, unlabeled_z.T) / TEMPERATURE
        
        # Create mask for positive pairs (augmented versions of same sample)
        n_unlabeled = unlabeled_z_i.shape[0]
        pos_mask = torch.zeros((2*n_unlabeled, 2*n_unlabeled), device=device, dtype=torch.bool)
        
        # Mark positive pairs (augmented versions of same sample)
        for i in range(n_unlabeled):
            pos_mask[i, n_unlabeled + i] = True
            pos_mask[n_unlabeled + i, i] = True
        
        # Remove self-similarity
        sim_matrix.fill_diagonal_(float('-inf'))
        
        # Compute loss
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()  # For numerical stability
        
        # For each anchor, compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean of positive pair losses
        mean_log_prob = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        unlabeled_loss = -mean_log_prob.mean()
        
        # Add to total loss
        loss += unlabeled_loss
    
    return loss

# Simplified version of contrastive loss for pretraining
def simclr_contrastive_loss(z_i, z_j):
    """
    SimCLR contrastive loss function
    
    Args:
        z_i: Embeddings from first augmentation
        z_j: Embeddings from second augmentation
    
    Returns:
        Loss value
    """
    batch_size = z_i.shape[0]
    device = z_i.device
    
    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate embeddings from both augmentations
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / TEMPERATURE
    
    # Create mask for positive pairs
    pos_mask = torch.zeros((2*batch_size, 2*batch_size), device=device, dtype=torch.bool)
    
    # Mark positive pairs (augmented versions of same sample)
    for i in range(batch_size):
        pos_mask[i, batch_size + i] = True
        pos_mask[batch_size + i, i] = True
    
    # Remove self-similarity
    sim_matrix.fill_diagonal_(float('-inf'))
    
    # Compute loss
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()  # For numerical stability
    
    # For each anchor, compute loss
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Compute mean of positive pair losses
    mean_log_prob = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    loss = -mean_log_prob.mean()
    
    return loss

# Training function for semi-supervised learning
def train_semi_supervised(pose_encoder, text_encoder, pose_projector, text_projector, 
                          labeled_loader, unlabeled_loader, tokenizer, optimizer, 
                          epochs=10, device=torch.device("cpu"), verbose=1):
    """
    Train the encoders and projectors using semi-supervised contrastive learning
    
    Args:
        pose_encoder: The pose encoder model
        text_encoder: The text encoder model
        pose_projector: Projector head for pose embeddings
        text_projector: Projector head for text embeddings
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled data
        tokenizer: BERT tokenizer for text processing
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to use for training
        verbose: Verbosity level
    """
    import time
    from datetime import timedelta
    
    # Set models to training mode
    pose_encoder.train()
    text_encoder.train()
    pose_projector.train()
    text_projector.train()
    
    # Print training configuration if verbose
    if verbose > 0:
        print(f"\n{'='*50}")
        print(f"Starting semi-supervised training with {epochs} epochs")
        print(f"Device: {device}")
        print(f"Labeled dataset size: {len(labeled_loader.dataset)} samples")
        print(f"Unlabeled dataset size: {len(unlabeled_loader.dataset)} samples")
        print(f"{'='*50}\n")
    
    # Track overall training time
    training_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0
        
        # Create iterators for both loaders
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Determine number of batches
        n_labeled_batches = len(labeled_loader)
        n_unlabeled_batches = len(unlabeled_loader)
        n_batches = max(n_labeled_batches, n_unlabeled_batches)
        
        # Print epoch header if verbose
        if verbose > 0:
            print(f"Epoch {epoch+1}/{epochs} - Started")
        
        # Iterate through batches
        for batch_idx in range(n_batches):
            batch_start = time.time()
            
            # Get labeled batch (if available)
            try:
                labeled_batch = next(labeled_iter)
                poses, texts = labeled_batch
                has_labeled = True
            except StopIteration:
                has_labeled = False
            
            # Get unlabeled batch (if available)
            try:
                unlabeled_batch = next(unlabeled_iter)
                unlabeled_poses = unlabeled_batch
                has_unlabeled = True
            except StopIteration:
                has_unlabeled = False
            
            # Skip if no data
            if not has_labeled and not has_unlabeled:
                continue
            
            # Process labeled data
            if has_labeled:
                # Move data to device
                poses = poses.to(device).float()
                encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                
                # Forward pass through encoders
                pose_embeds = pose_encoder(poses)
                text_embeds = text_encoder(input_ids, attention_mask)
                
                # Forward pass through projectors
                pose_projections = pose_projector(pose_embeds)
                text_projections = text_projector(text_embeds)
                
                # Calculate supervised contrastive loss
                loss = simclr_contrastive_loss(pose_projections, text_projections)
            
            # Process unlabeled data
            if has_unlabeled:
                # Move data to device
                unlabeled_poses = unlabeled_poses.to(device).float()
                
                # Create augmented versions
                augmented_poses = torch.stack([torch.tensor(augment_sequence(p.cpu().numpy())) 
                                             for p in unlabeled_poses]).to(device).float()
                
                # Forward pass through encoder and projector
                unlabeled_embeds = pose_encoder(unlabeled_poses)
                augmented_embeds = pose_encoder(augmented_poses)
                
                # Forward pass through projector
                unlabeled_projections = pose_projector(unlabeled_embeds)
                augmented_projections = pose_projector(augmented_embeds)
                
                # Calculate unsupervised contrastive loss
                unsupervised_loss = simclr_contrastive_loss(unlabeled_projections, augmented_projections)
                
                # Add to total loss
                if has_labeled:
                    loss = 0.5 * (loss + unsupervised_loss)
                else:
                    loss = unsupervised_loss
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            loss_val = loss.item()
            total_loss += loss_val
            batch_count += 1
            
            # Print batch progress if verbose level 2
            if verbose > 1:
                batch_time = time.time() - batch_start
                print(f"  Batch {batch_idx+1}/{n_batches} - "
                      f"Loss: {loss_val:.4f} - "
                      f"Time: {batch_time:.2f}s")
        
        # Calculate epoch statistics
        epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        if verbose > 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Time: {timedelta(seconds=int(epoch_time))}")
    
    # Print training summary
    if verbose > 0:
        total_time = time.time() - training_start
        print(f"\n{'='*50}")
        print(f"Training completed in {timedelta(seconds=int(total_time))}")
        print(f"Final loss: {epoch_loss:.4f}")
        print(f"{'='*50}\n")

# Function to train classifier on frozen features
def train_classifier(pose_encoder, classifier, train_loader, val_loader=None, 
                     epochs=10, device=torch.device("cpu"), verbose=1):
    """
    Train a classifier on top of frozen encoder
    
    Args:
        pose_encoder: The frozen pose encoder model
        classifier: The classifier model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        epochs: Number of training epochs
        device: Device to use for training
        verbose: Verbosity level
    """
    import time
    from datetime import timedelta
    
    # Set models to appropriate modes
    pose_encoder.eval()  # Freeze encoder
    classifier.train()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Print training configuration if verbose
    if verbose > 0:
        print(f"\n{'='*50}")
        print(f"Training classifier with {epochs} epochs")
        print(f"Device: {device}")
        print(f"Dataset size: {len(train_loader.dataset)} samples")
        print(f"{'='*50}\n")
    
    # Track overall training time
    training_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Print epoch header if verbose
        if verbose > 0:
            print(f"Epoch {epoch+1}/{epochs} - Started")
        
        # Training loop
        classifier.train()
        for batch_idx, (poses, labels) in enumerate(train_loader):
            batch_start = time.time()
            
            # Move data to device
            poses = poses.to(device).float()
            labels = labels.to(device).long().squeeze()
            
            # Forward pass through frozen encoder
            with torch.no_grad():
                features = pose_encoder(poses)
            
            # Forward pass through classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Print batch progress if verbose level 2
            if verbose > 1 and (batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1):
                batch_time = time.time() - batch_start
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - "
                      f"Acc: {100.*train_correct/train_total:.2f}% - "
                      f"Time: {batch_time:.2f}s")
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation loop
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if val_loader is not None:
            classifier.eval()
            with torch.no_grad():
                for poses, labels in val_loader:
                    poses = poses.to(device).float()
                    labels = labels.to(device).long().squeeze()
                    
                    # Forward pass through frozen encoder
                    features = pose_encoder(poses)
                    
                    # Forward pass through classifier
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    
                    # Track metrics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        if verbose > 0:
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Train Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_acc:.2f}% - "
                      f"Time: {timedelta(seconds=int(epoch_time))}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Train Acc: {train_acc:.2f}% - "
                      f"Time: {timedelta(seconds=int(epoch_time))}")
    
    # Print training summary
    if verbose > 0:
        total_time = time.time() - training_start
        print(f"\n{'='*50}")
        print(f"Classifier training completed in {timedelta(seconds=int(total_time))}")
        print(f"Final train accuracy: {train_acc:.2f}%")
        if val_loader is not None:
            print(f"Final validation accuracy: {val_acc:.2f}%")
        print(f"{'='*50}\n")
    
    return train_acc, val_acc if val_loader is not None else None