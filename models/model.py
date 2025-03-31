import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import random

# ======================= CONFIG ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 40
INPUT_DIM = 159
TEXT_EMBED_DIM = 768
POSE_EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 80
TEMPERATURE = 0.07

# ================== TEXT ENCODER ==========================

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(TEXT_EMBED_DIM, POSE_EMBED_DIM)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        return self.linear(pooled_output)  # [batch_size, 256]

# ================== POSE ENCODER ==========================

class PoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.linear = nn.Linear(HIDDEN_DIM, POSE_EMBED_DIM)

    def forward(self, x):  # x: [batch_size, seq_len, input_dim]
        _, (hidden, _) = self.lstm(x)  # hidden: [1, batch_size, hidden_dim]
        return self.linear(hidden.squeeze(0))  # [batch_size, 256]

# =================== CONTRASTIVE LOSS =====================

def contrastive_loss(text_embeds, pose_embeds):
    text_norm = F.normalize(text_embeds, dim=1)
    pose_norm = F.normalize(pose_embeds, dim=1)
    logits = torch.matmul(text_norm, pose_norm.T) / TEMPERATURE
    labels = torch.arange(len(logits)).to(DEVICE)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

# ================== MAIN TRAINING ==========================

def train(text_encoder, pose_encoder, data_loader, tokenizer, optimizer, epochs=10, verbose=1):
    """
    Train the text and pose encoders using contrastive learning.
    
    Args:
        text_encoder: The text encoder model
        pose_encoder: The pose encoder model
        data_loader: DataLoader containing pose-text pairs
        tokenizer: BERT tokenizer for text processing
        optimizer: Optimizer for training
        epochs: Number of training epochs
        verbose: Verbosity level (0=silent, 1=epoch summaries, 2=batch updates)
    """
    import time
    from datetime import timedelta
    
    text_encoder.train()
    pose_encoder.train()
    
    # Print training configuration if verbose
    if verbose > 0:
        print(f"\n{'='*50}")
        print(f"Starting training with {epochs} epochs")
        print(f"Device: {DEVICE}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Dataset size: {len(data_loader.dataset)} samples")
        print(f"Steps per epoch: {len(data_loader)}")
        print(f"{'='*50}\n")
    
    # Track overall training time
    training_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        batch_losses = []
        
        # Print epoch header if verbose
        if verbose > 0:
            print(f"Epoch {epoch+1}/{epochs} - Started")
        
        # Iterate through batches
        for batch_idx, (poses, texts) in enumerate(data_loader):
            batch_start = time.time()
            
            # Move data to device
            poses = poses.to(DEVICE).float()
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)
            
            # Forward pass
            text_embeds = text_encoder(input_ids, attention_mask)
            pose_embeds = pose_encoder(poses)
            
            # Calculate loss and update weights
            loss = contrastive_loss(text_embeds, pose_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            loss_val = loss.item()
            total_loss += loss_val
            batch_losses.append(loss_val)
            
            # Print batch progress if verbose level 2
            if verbose > 1:
                batch_time = time.time() - batch_start
                print(f"  Batch {batch_idx+1}/{len(data_loader)} - "
                      f"Loss: {loss_val:.4f} - "
                      f"Time: {batch_time:.2f}s")
        
        # Calculate epoch statistics
        epoch_loss = total_loss / len(data_loader)
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        if verbose > 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Min Loss: {min(batch_losses):.4f} - "
                  f"Max Loss: {max(batch_losses):.4f} - "
                  f"Time: {timedelta(seconds=int(epoch_time))}")
    
    # Print training summary
    if verbose > 0:
        total_time = time.time() - training_start
        print(f"\n{'='*50}")
        print(f"Training completed in {timedelta(seconds=int(total_time))}")
        print(f"Final loss: {epoch_loss:.4f}")
        print(f"{'='*50}\n")

# ========== EXAMPLES: TEXT TO DANCE / DANCE TO TEXT ========

def generate_dance(text_encoder, pose_encoder, tokenizer, text_query, pose_dataset):
    text_encoder.eval()
    pose_encoder.eval()
    with torch.no_grad():
        encoded = tokenizer(text_query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        text_embed = text_encoder(encoded["input_ids"], encoded["attention_mask"]).squeeze(0)
        min_dist = float('inf')
        best_pose = None
        for pose_seq in pose_dataset:
            pose_seq = torch.tensor(pose_seq).unsqueeze(0).to(DEVICE)
            pose_embed = pose_encoder(pose_seq).squeeze(0)
            dist = torch.norm(text_embed - pose_embed).item()
            if dist < min_dist:
                min_dist = dist
                best_pose = pose_seq
        return best_pose.cpu().numpy()

def generate_text(text_encoder, pose_encoder, tokenizer, pose_query, text_candidates):
    text_encoder.eval()
    pose_encoder.eval()
    with torch.no_grad():
        pose_query = torch.tensor(pose_query).unsqueeze(0).to(DEVICE)
        pose_embed = pose_encoder(pose_query).squeeze(0)
        best_score = -float('inf')
        best_text = None
        for text in text_candidates:
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            text_embed = text_encoder(encoded["input_ids"], encoded["attention_mask"]).squeeze(0)
            score = torch.dot(F.normalize(text_embed, dim=0), F.normalize(pose_embed, dim=0)).item()
            if score > best_score:
                best_score = score
                best_text = text
        return best_text

# ========== PUTTING IT TOGETHER ============================

class DanceTextDataset(torch.utils.data.Dataset):
    def __init__(self, pose_data, text_data):
        self.pose_data = pose_data
        self.text_data = text_data

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        return self.pose_data[idx], self.text_data[idx]

# You'd initialize your data, model, and train like this:
# pose_encoder = PoseEncoder().to(DEVICE)
# text_encoder = TextEncoder().to(DEVICE)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# optimizer = torch.optim.Adam(list(pose_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)
# dataset = DanceTextDataset(labelled_pose_tensor, effort_text_labels["time_and_space"])
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# train(text_encoder, pose_encoder, loader, tokenizer, optimizer)

# Then test with:
# generate_dance(text_encoder, pose_encoder, tokenizer, "a fast energetic leap", test_pose_dataset)
# generate_text(text_encoder, pose_encoder, tokenizer, some_pose_seq, ["a slow turn", "a fast jump", ...])