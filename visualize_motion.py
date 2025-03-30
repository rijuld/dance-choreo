
import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import MarielDataset
from utils.plotting import animate_stick
from IPython.display import HTML


ffmpeg_path='/opt/homebrew/bin/ffmpeg' # point to your specific copy of the ffmpeg binary
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path # for using html5 video in Jupyter notebook


# Initialize the dataset
dataset = MarielDataset(
    reduced_joints=False,  # Use all joints
    xy_centering=True,    # Center the motion in XY plane
    seq_len=128,          # Length of sequence to visualize
    file_path="data/mariel_beyond.npy"  # Example motion file
)

# Get a sample sequence
sample = dataset[0]  # Get first sequence
sequence = sample.x.numpy()  # Convert to numpy array
print(sequence.shape)
# Reshape the sequence back to (seq_len, n_joints, n_dim)
seq_len = 128
n_joints = 53
n_dim = 6
sequence = sequence.reshape((n_joints, seq_len, n_dim))
sequence = np.transpose(sequence, [1, 0, 2])  # Put seq_len first
sequence = sequence[:, :, :3]  # Take only x, y, z

print(sequence.shape)
# Create animation with different visualization options
anim = animate_stick(
    sequence,
    figsize=(10, 10),
    dot_size=30,
    dot_alpha=0.8,
    skeleton=True,         # Show skeleton lines
    skeleton_alpha=0.5,    # Skeleton line opacity
    cloud=True,            # Show point cloud connections
    cloud_alpha=0.1,       # Cloud connection opacity
    ax_lims=(-1, 1),      # Set axis limits
    speed=45              # Animation speed
)

# Display the animation
HTML(anim.to_html5_video())