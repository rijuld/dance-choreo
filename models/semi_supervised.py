import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
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
    
    # Generate all random decisions at once for better performance
    # apply_augmentations = np.random.random(3) > 0.5
    apply_augmentations = [False, False, True]
    
    # Apply augmentations based on random decisions
    if apply_augmentations[0]:
        augmented = time_crop(augmented)
    if apply_augmentations[1]:
        augmented = jitter(augmented)
    if apply_augmentations[2]:
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
def semi_supervised_contrastive_loss(z_i, z_j=None, labels=None, mask=None, temperature=TEMPERATURE):
    """
    Compute contrastive loss for semi-supervised learning with enhanced numerical stability
    
    Args:
        z_i: Embeddings from encoder+projector
        z_j: Optional augmented embeddings (for unlabeled data)
        labels: Optional labels (for labeled data)
        mask: Boolean mask indicating which samples are labeled
        temperature: Temperature parameter for scaling similarity scores (default: TEMPERATURE)
    
    Returns:
        Loss value
    """
    # Check for invalid inputs and apply preprocessing
    if z_i is None:
        print("ERROR: Input embeddings (z_i) cannot be None")
        return torch.tensor(0.0, device=torch.device("cpu"), requires_grad=True)
        
    # Handle NaN and Inf values in input embeddings
    if torch.isnan(z_i).any() or torch.isinf(z_i).any():
        print("WARNING: NaN or Inf values detected in input embeddings (z_i)")
        # Replace NaN/Inf values with zeros
        z_i = torch.nan_to_num(z_i, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if z_j is not None and (torch.isnan(z_j).any() or torch.isinf(z_j).any()):
        print("WARNING: NaN or Inf values detected in input embeddings (z_j)")
        # Replace NaN/Inf values with zeros
        z_j = torch.nan_to_num(z_j, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Get device and batch size
    device = z_i.device
    batch_size = z_i.shape[0]
    
    # Check for empty batch
    if batch_size == 0:
        print("WARNING: Empty batch detected")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalize embeddings (with epsilon to prevent division by zero)
    z_i_norm = torch.norm(z_i, p=2, dim=1, keepdim=True)
    z_i_norm = torch.clamp(z_i_norm, min=1e-8)  # Prevent division by zero
    z_i = z_i / z_i_norm
    
    if z_j is not None:
        z_j_norm = torch.norm(z_j, p=2, dim=1, keepdim=True)
        z_j_norm = torch.clamp(z_j_norm, min=1e-8)  # Prevent division by zero
        z_j = z_j / z_j_norm
    
    # Initialize loss
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Process labeled data (supervised contrastive loss)
    if labels is not None and mask is not None and torch.any(mask):
        labeled_z = z_i[mask]
        labeled_labels = labels[mask]
        
        # Check if we have enough labeled samples
        n_labeled = labeled_z.shape[0]
        if n_labeled <= 1:
            print("WARNING: Not enough labeled samples for contrastive loss")
        else:
            # Compute similarity matrix for labeled data with temperature scaling
            sim_matrix = torch.matmul(labeled_z, labeled_z.T) / temperature
            
            # Create mask for positive pairs (same label)
            label_mask = labeled_labels.unsqueeze(1) == labeled_labels.unsqueeze(0)
            # Remove self-similarity
            label_mask.fill_diagonal_(False)
            
            # For each anchor, compute loss over positive pairs
            labeled_loss_sum = 0.0
            valid_samples = 0
            
            for i in range(n_labeled):
                # Check if there are any positive pairs for this sample
                if not torch.any(label_mask[i]):
                    continue  # Skip if no positive pairs
                
                # Get positive and negative pairs
                pos_mask_i = label_mask[i]
                neg_mask_i = ~pos_mask_i
                neg_mask_i[i] = False  # Remove self
                
                # Skip if no negative pairs
                if not torch.any(neg_mask_i):
                    continue
                
                # Get similarity scores
                pos_sim = sim_matrix[i][pos_mask_i]
                neg_sim = sim_matrix[i][neg_mask_i]
                
                # Compute logits with improved numerical stability
                logits = torch.cat([pos_sim, neg_sim])
                
                # Apply max subtraction for numerical stability
                logits_max = torch.max(logits)
                logits = logits - logits_max.detach()
                
                # Create target distribution (uniform over positive pairs)
                n_pos = pos_sim.shape[0]
                labels_i = torch.zeros(logits.shape[0], device=device)
                labels_i[:n_pos] = 1.0 / n_pos
                
                # Compute softmax with improved numerical stability
                exp_logits = torch.exp(logits)
                exp_sum = torch.sum(exp_logits) + 1e-10  # Add epsilon to prevent division by zero
                log_probs = logits - torch.log(exp_sum)
                
                # Compute loss
                if not torch.isnan(log_probs).any() and not torch.isinf(log_probs).any():
                    loss_i = -torch.sum(labels_i * log_probs)
                    labeled_loss_sum += loss_i
                    valid_samples += 1
                else:
                    print(f"WARNING: NaN or Inf values in log_probs for sample {i}")
            
            # Normalize by number of valid samples
            if valid_samples > 0:
                loss = labeled_loss_sum / valid_samples
            else:
                print("WARNING: No valid samples for labeled loss calculation")
    
    # Process unlabeled data (SimCLR-style contrastive loss)
    if z_j is not None and mask is not None and torch.any(~mask):
        unlabeled_z_i = z_i[~mask]
        unlabeled_z_j = z_j[~mask]
        
        # Check if we have enough unlabeled samples
        n_unlabeled = unlabeled_z_i.shape[0]
        if n_unlabeled == 0:
            print("WARNING: No unlabeled samples for contrastive loss")
        else:
            # Concatenate embeddings from both augmentations
            unlabeled_z = torch.cat([unlabeled_z_i, unlabeled_z_j], dim=0)
            
            # Compute similarity matrix with temperature scaling
            sim_matrix = torch.matmul(unlabeled_z, unlabeled_z.T) / temperature
            
            # Create mask for positive pairs (augmented versions of same sample)
            pos_mask = torch.zeros((2*n_unlabeled, 2*n_unlabeled), device=device, dtype=torch.bool)
            
            # Mark positive pairs (augmented versions of same sample)
            for i in range(n_unlabeled):
                pos_mask[i, n_unlabeled + i] = True
                pos_mask[n_unlabeled + i, i] = True
            
            # Remove self-similarity
            sim_matrix.fill_diagonal_(float('-inf'))
            
            # Compute loss with enhanced numerical stability
            # Apply max subtraction for numerical stability (per row)
            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - logits_max.detach()
            
            # For each anchor, compute loss
            exp_sim = torch.exp(sim_matrix)
            # Add small epsilon to prevent log(0) and division by zero
            exp_sum = exp_sim.sum(dim=1, keepdim=True)
            exp_sum = torch.clamp(exp_sum, min=1e-10)
            log_prob = sim_matrix - torch.log(exp_sum)
            
            # Ensure we don't divide by zero
            pos_sum = pos_mask.sum(1)
            pos_sum = torch.clamp(pos_sum, min=1)  # Ensure at least 1 positive pair
            
            # Compute mean of positive pair losses
            mean_log_prob = (pos_mask * log_prob).sum(1) / pos_sum
            
            # Check for NaN or Inf values
            if torch.isnan(mean_log_prob).any() or torch.isinf(mean_log_prob).any():
                print("WARNING: NaN or Inf values detected in unlabeled log probabilities")
                mean_log_prob = torch.nan_to_num(mean_log_prob, nan=0.0, posinf=0.0, neginf=0.0)
            
            unlabeled_loss = -mean_log_prob.mean()
            
            # Add to total loss if valid
            if not torch.isnan(unlabeled_loss) and not torch.isinf(unlabeled_loss):
                # If we already have a labeled loss, average them
                if loss > 0:
                    loss = 0.5 * (loss + unlabeled_loss)
                else:
                    loss = unlabeled_loss
            else:
                print("WARNING: Unlabeled loss is NaN or Inf, skipping")
    
    # Final safety check
    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: Final loss is NaN or Inf, returning zero loss")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss

# Simplified version of contrastive loss for pretraining
def simclr_contrastive_loss(z_i, z_j, temperature=TEMPERATURE):
    """
    More numerically stable implementation of SimCLR contrastive loss
    """
    # Input validation with better error messages
    if z_i is None or z_j is None:
        print("ERROR: Input embeddings cannot be None")
        return torch.tensor(0.0, device=torch.device("cpu"), requires_grad=True)
    
    # Get batch size and device
    batch_size = z_i.shape[0]
    device = z_i.device
    
    if batch_size == 0:
        print("WARNING: Empty batch detected")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 1. More robust normalization with epsilon
    epsilon = 1e-8
    z_i_norm = torch.norm(z_i, p=2, dim=1, keepdim=True).clamp(min=epsilon)
    z_j_norm = torch.norm(z_j, p=2, dim=1, keepdim=True).clamp(min=epsilon)
    
    z_i = z_i / z_i_norm
    z_j = z_j / z_j_norm
    
    # 2. Pre-check for NaN/Inf after normalization
    if torch.isnan(z_i).any() or torch.isinf(z_i).any() or torch.isnan(z_j).any() or torch.isinf(z_j).any():
        print("WARNING: NaN/Inf values after normalization, applying nan_to_num")
        z_i = torch.nan_to_num(z_i, nan=0.0, posinf=0.0, neginf=0.0)
        z_j = torch.nan_to_num(z_j, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Concatenate embeddings
    z = torch.cat([z_i, z_j], dim=0)
    
    # 3. Use a more stable temperature value
    actual_temp = max(temperature, 0.01)  # Prevent temperature from being too small
    
    # 4. More stable similarity computation
    sim_matrix = torch.mm(z, z.t()) / actual_temp
    
    # 5. Create mask for positive pairs more efficiently
    mask = torch.zeros((2*batch_size, 2*batch_size), device=device, dtype=torch.bool)
    for i in range(batch_size):
        mask[i, batch_size + i] = True
        mask[batch_size + i, i] = True
    
    # Remove self-similarity with a large negative value instead of -inf
    sim_matrix.fill_diagonal_(-9999.0)
    
    # 6. Apply logsumexp for improved numerical stability
    # First, get the maximum value for each row for numerical stability
    max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    
    # Subtract max before exponentiating (prevents overflow)
    exp_sim = torch.exp(sim_matrix - max_sim)
    
    # 7. Add epsilon to prevent log(0)
    log_prob = sim_matrix - max_sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + epsilon)
    
    # 8. Handle mask more carefully
    pos_mask_sum = mask.sum(dim=1)
    # Ensure there's at least one positive pair per anchor
    if torch.any(pos_mask_sum == 0):
        print("WARNING: Some samples have no positive pairs")
        pos_mask_sum = torch.clamp(pos_mask_sum, min=1)
    
    # Compute mean of positive pair log probabilities
    mean_log_prob = (mask * log_prob).sum(dim=1) / pos_mask_sum
    
    # 9. Check for NaN/Inf and handle before computing final loss
    if torch.isnan(mean_log_prob).any() or torch.isinf(mean_log_prob).any():
        print("WARNING: NaN or Inf values in log probabilities, replacing with zeros")
        mean_log_prob = torch.nan_to_num(mean_log_prob, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute final loss
    loss = -mean_log_prob.mean()
    
    # 10. Final safety check
    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: Final loss is NaN or Inf, returning zero loss")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss

# Training function for semi-supervised learning
def train_semi_supervised(pose_encoder, text_encoder, pose_projector, text_projector, 
                          labeled_loader, unlabeled_loader, tokenizer, optimizer, 
                          epochs=10, device=torch.device("cpu"), verbose=1,
                          checkpoint_dir=None, start_epoch=0, training_history=None,
                          text_labels=None):
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
    
    # Initialize training history if not provided
    if training_history is None:
        training_history = []
    
    # Check for existing checkpoint if directory is provided
    if checkpoint_dir is not None and start_epoch == 0:  # Only check if not already resuming
        checkpoint_path = os.path.join(checkpoint_dir, "semi_supervised_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            if verbose > 0:
                print(f"\nFound existing checkpoint at {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            pose_encoder.load_state_dict(checkpoint['pose_encoder'])
            text_encoder.load_state_dict(checkpoint['text_encoder'])
            pose_projector.load_state_dict(checkpoint['pose_projector'])
            text_projector.load_state_dict(checkpoint['text_projector'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            training_history = checkpoint.get('history', [])
            if verbose > 0:
                print(f"Resuming training from epoch {start_epoch}")
                if 'train_acc' in checkpoint:
                    print(f"Previous training accuracy: {checkpoint['train_acc']:.2f}%")
                if 'val_acc' in checkpoint and checkpoint['val_acc'] is not None:
                    print(f"Previous validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    for epoch in range(start_epoch, start_epoch + epochs):
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
                poses, indices = labeled_batch
                # Get actual text labels using indices
                if 'text_labels' in locals() or 'text_labels' in globals():
                    texts = [text_labels[idx.item()] for idx in indices]
                else:
                    # Fallback if text_labels not provided
                    texts = [f"Label {idx.item()}" for idx in indices]
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
                # Process text labels based on their type
                if text_labels is not None:
                    # Check if texts already contains strings (from earlier processing)
                    if isinstance(texts[0], str):
                        # Already strings, use directly
                        text_strings = texts
                    else:
                        # Convert indices to list of strings
                        text_strings = [text_labels[idx.item()] for idx in texts]
                        print(text_strings)
                    encoded = tokenizer(text_strings, return_tensors="pt", padding=True, truncation=True)
                else:
                    # Fallback if text_labels not provided (should not happen)
                    if isinstance(texts[0], str):
                        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                    else:
                        encoded = tokenizer([f"Label {idx.item()}" for idx in texts], return_tensors="pt", padding=True, truncation=True)
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
                
                # Create augmented versions - optimized to reduce CPU-GPU transfers
                # Process in batches on CPU and then transfer to GPU once
                with torch.no_grad():
                    # Move to CPU for augmentation
                    unlabeled_poses_cpu = unlabeled_poses.cpu()
                    batch_size = unlabeled_poses_cpu.size(0)
                    augmented_poses_list = []
                    
                    # Process in smaller batches to avoid memory issues
                    sub_batch_size = 8  # Adjust based on memory constraints
                    for i in range(0, batch_size, sub_batch_size):
                        end_idx = min(i + sub_batch_size, batch_size)
                        sub_batch = unlabeled_poses_cpu[i:end_idx]
                        
                        # Apply augmentation to each sequence in the sub-batch
                        augmented_sub_batch = torch.stack([
                            torch.tensor(augment_sequence(p.numpy())) 
                            for p in sub_batch
                        ])
                        
                        augmented_poses_list.append(augmented_sub_batch)
                    
                    # Combine all sub-batches and move to device
                    augmented_poses = torch.cat(augmented_poses_list, dim=0).to(device).float()
                
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
            
            # Update weights with gradient clipping to prevent exploding gradients
            optimizer.zero_grad()
            
            # Check if loss is valid before backpropagation
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN or Inf loss detected before backward pass, skipping update")
                # Skip this batch entirely
                continue
                
            # Backward pass
            loss.backward()
            
            # Enhanced gradient clipping to prevent exploding gradients
            # Only clip the classifier parameters since we're only training the classifier
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            # Comprehensive check for NaN/Inf gradients
            has_invalid_grad = False
            for param in classifier.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_invalid_grad = True
                        # Replace NaN/Inf gradients with zeros to allow training to continue
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            if has_invalid_grad:
                print("WARNING: NaN or Inf gradients detected and replaced with zeros")
            
            # Apply gradients
            optimizer.step()
            
            # Verify model parameters are valid after update
            has_invalid_param = False
            for param in classifier.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_invalid_param = True
                    break
            
            if has_invalid_param:
                print("WARNING: NaN or Inf values detected in model parameters after update")
                # Could implement parameter restoration from previous checkpoint here if needed
            
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
            print(f"Epoch {epoch+1}/{(start_epoch + epochs)} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Time: {timedelta(seconds=int(epoch_time))}")
        
        # Save epoch metrics to history
        epoch_metrics = {
            'epoch': epoch,
            'loss': epoch_loss,
            'time': epoch_time
        }
        training_history.append(epoch_metrics)
        
        # Save checkpoint if directory is provided
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "semi_supervised_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'pose_encoder': pose_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'pose_projector': pose_projector.state_dict(),
                'text_projector': text_projector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': epoch_loss,
                'history': training_history
            }, checkpoint_path)
            
            # Save training history as JSON
            history_path = os.path.join(checkpoint_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
    


# Function to train classifier on frozen features
def train_classifier(pose_encoder, classifier, train_loader, val_loader=None, 
                     epochs=10, device=torch.device("cpu"), verbose=1,
                     checkpoint_dir=None, start_epoch=0, training_history=None):
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
    
    # Initialize training history if not provided
    if training_history is None:
        training_history = []
        
    # Check for existing checkpoint if directory is provided
    if checkpoint_dir is not None and start_epoch == 0:  # Only check if not already resuming
        checkpoint_path = os.path.join(checkpoint_dir, "classifier_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            if verbose > 0:
                print(f"\nFound existing checkpoint at {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            training_history = checkpoint.get('history', [])
            if verbose > 0:
                print(f"Resuming training from epoch {start_epoch}")
                if 'train_acc' in checkpoint:
                    print(f"Previous training accuracy: {checkpoint['train_acc']:.2f}%")
                if 'val_acc' in checkpoint and checkpoint['val_acc'] is not None:
                    print(f"Previous validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    for epoch in range(start_epoch, start_epoch + epochs):
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
            
            # Update weights with gradient clipping to prevent exploding gradients
            optimizer.zero_grad()
            
            # Check if loss is valid before backpropagation
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN or Inf loss detected before backward pass, skipping update")
                # Skip this batch entirely
                continue
                
            # Backward pass
            loss.backward()
            
            # Enhanced gradient clipping to prevent exploding gradients
            # Only clip the classifier parameters since we're only training the classifier
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            # Comprehensive check for NaN/Inf gradients
            has_invalid_grad = False
            for param in classifier.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_invalid_grad = True
                        # Replace NaN/Inf gradients with zeros to allow training to continue
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            if has_invalid_grad:
                print("WARNING: NaN or Inf gradients detected and replaced with zeros")
            
            # Apply gradients
            optimizer.step()
            
            # Verify model parameters are valid after update
            has_invalid_param = False
            for param in classifier.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_invalid_param = True
                    break
            
            if has_invalid_param:
                print("WARNING: NaN or Inf values detected in model parameters after update")
                # Could implement parameter restoration from previous checkpoint here if needed
            
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
        
        # Save epoch metrics to history
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'time': epoch_time
        }
        
        if val_loader is not None:
            epoch_metrics['val_loss'] = val_loss
            epoch_metrics['val_acc'] = val_acc
            
        training_history.append(epoch_metrics)
        
        # Save checkpoint if directory is provided
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "classifier_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'pose_encoder': pose_encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc if val_loader is not None else None,
                'history': training_history
            }, checkpoint_path)
            
            # Save training history as JSON
            history_path = os.path.join(checkpoint_dir, "classifier_training_history.json")
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
    
    # Print training summary
    if verbose > 0:
        total_time = time.time() - training_start
        print(f"\n{'='*50}")
        print(f"Classifier training completed in {timedelta(seconds=int(total_time))}")
        print(f"Final train accuracy: {train_acc:.2f}%")
        if val_loader is not None:
            print(f"Final validation accuracy: {val_acc:.2f}%")
        print(f"{'='*50}\n")
    
    # Save final checkpoint if directory is provided
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, "classifier_checkpoint.pt")
        torch.save({
            'epoch': epoch,
            'pose_encoder': pose_encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc if val_loader is not None else None,
            'history': training_history
        }, checkpoint_path)
        
        # Save training history as JSON
        history_path = os.path.join(checkpoint_dir, "classifier_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    return train_acc, val_acc if val_loader is not None else None