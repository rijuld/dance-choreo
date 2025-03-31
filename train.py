import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

from utils import datasets
from utils.config import default_config
from models.model import TextEncoder, PoseEncoder, train, DanceTextDataset, DEVICE, BATCH_SIZE

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
    print("Loading data...")
    # Load both datasets
    config_time = get_config("time")
    config_space = get_config("space")
    
    # Load with same strategy (choose one)
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
    
    print(f"Prepared {len(combined_text_labels)} text-pose pairs for training")
    
    # Initialize model components
    print("Initializing models...")
    pose_encoder = PoseEncoder().to(DEVICE)
    text_encoder = TextEncoder().to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.Adam(
        list(pose_encoder.parameters()) + list(text_encoder.parameters()), 
        lr=1e-4
    )
    
    # Create dataset and dataloader
    dataset = DanceTextDataset(pose_tensor, combined_text_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train the model
    print("Starting training...")
    train(text_encoder, pose_encoder, loader, tokenizer, optimizer, epochs=10, verbose=2)
    
    # Save the trained models
    print("Saving models...")
    torch.save(text_encoder.state_dict(), "text_encoder.pth")
    torch.save(pose_encoder.state_dict(), "pose_encoder.pth")
    
    print("Training complete!")

if __name__ == "__main__":
    main()