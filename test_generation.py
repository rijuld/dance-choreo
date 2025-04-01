import torch
import numpy as np
from models.model import TextEncoder, PoseEncoder
from transformers import BertTokenizer
from utils.datasets import load_raw, sequify_all_data
from utils.config import default_config

def load_models(text_encoder_path="text_encoder_semi.pth", pose_encoder_path="pose_encoder_semi.pth"):
    """Load pretrained text and pose encoders."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    text_encoder = TextEncoder().to(device)
    pose_encoder = PoseEncoder().to(device)
    
    # Load pretrained weights
    text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=device))
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=device))
    
    # Set to evaluation mode
    text_encoder.eval()
    pose_encoder.eval()
    
    return text_encoder, pose_encoder, device

def prepare_test_set(n_samples=10):
    """Prepare a small holdout test set of dance sequences."""
    # Load raw data
    all_data, all_data_centered = load_raw()
    
    # Reshape data for sequification
    n_poses, n_joints, dims = all_data_centered.shape
    all_data_reshaped = all_data_centered.reshape(n_poses, n_joints * dims)
    
    # Create sequences
    sequences = sequify_all_data(all_data_reshaped, default_config.seq_len, augmentation_factor=1)
    
    # Randomly select n_samples sequences for testing
    np.random.seed(42)
    test_indices = np.random.choice(len(sequences), n_samples, replace=False)
    test_sequences = sequences[test_indices]
    
    return torch.tensor(test_sequences, dtype=torch.float32)

def text_to_dance(text_query, text_encoder, pose_encoder, test_sequences, device, tokenizer):
    """Generate a dance sequence from a text description."""
    text_encoder.eval()
    pose_encoder.eval()
    
    with torch.no_grad():
        # Encode text query
        encoded = tokenizer(text_query, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embed = text_encoder(encoded["input_ids"], encoded["attention_mask"]).squeeze(0)
        
        # Find closest dance sequence
        min_dist = float('inf')
        best_pose = None
        
        for pose_seq in test_sequences:
            pose_seq = pose_seq.unsqueeze(0).to(device)
            pose_embed = pose_encoder(pose_seq).squeeze(0)
            dist = torch.norm(text_embed - pose_embed).item()
            
            if dist < min_dist:
                min_dist = dist
                best_pose = pose_seq
        
        return best_pose.cpu().numpy(), min_dist

def dance_to_text(pose_sequence, text_encoder, pose_encoder, device, tokenizer, text_candidates):
    """Generate a text description from a dance sequence."""
    text_encoder.eval()
    pose_encoder.eval()
    
    with torch.no_grad():
        # Encode pose sequence
        pose_query = torch.tensor(pose_sequence).unsqueeze(0).to(device)
        pose_embed = pose_encoder(pose_query).squeeze(0)
        
        # Find closest text description
        best_score = -float('inf')
        best_text = None
        scores = {}
        
        for text in text_candidates:
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embed = text_encoder(encoded["input_ids"], encoded["attention_mask"]).squeeze(0)
            score = torch.dot(torch.nn.functional.normalize(text_embed, dim=0),
                            torch.nn.functional.normalize(pose_embed, dim=0)).item()
            scores[text] = score
            
            if score > best_score:
                best_score = score
                best_text = text
        
        return best_text, scores

def main():
    # Load models and prepare test data
    text_encoder, pose_encoder, device = load_models()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_sequences = prepare_test_set(n_samples=10)
    
    # Example text descriptions for testing
    text_candidates = [
        "Quick and Direct movement",
        "Sustained and Indirect movement",
        "Quick and Indirect movement",
        "Sustained and Direct movement"
    ]
    
    print("\n=== Testing Text to Dance Generation ===")
    test_text = "Quick and Direct movement"
    print(f"\nInput text: {test_text}")
    best_sequence, distance = text_to_dance(
        test_text, text_encoder, pose_encoder, test_sequences, device, tokenizer
    )
    print(f"Generated dance sequence shape: {best_sequence.shape}")
    print(f"Distance score: {distance:.4f}")
    
    print("\n=== Testing Dance to Text Generation ===")
    # Use the first sequence from test set
    test_pose = test_sequences[0].numpy()
    print(f"\nInput dance sequence shape: {test_pose.shape}")
    best_text, scores = dance_to_text(
        test_pose, text_encoder, pose_encoder, device, tokenizer, text_candidates
    )
    print(f"Generated text description: {best_text}")
    print("\nScores for all candidates:")
    for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{text}: {score:.4f}")

if __name__ == "__main__":
    main()