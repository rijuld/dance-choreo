# Human-AI Choreography Project

## Overview
The Human-AI Choreography project explores the intersection of dance and artificial intelligence through the lens of Laban Movement Analysis. It uses semi-supervised learning to analyze and generate dance movements, creating a bridge between textual descriptions and physical motion.

## Project Structure
The project is organized into several components:

- **models/**: Contains model definitions for text and pose encoders, along with semi-supervised learning implementation
- **data/**: Stores motion capture data in .npy format from various dance sequences
- **label_generation/skelabel/**: Contains the PirouNet Labeler, a Dash-based web application for annotating dance motion capture sequences
- **utils/**: Includes utility functions for data processing, configuration, and visualization
- **training_dataset/**: Contains labeled datasets used for training
- **checkpoints/**: Stores model checkpoints during training

## Features

- **Motion-Text Mapping**: Bidirectional translation between dance movements and textual descriptions
- **Laban Movement Analysis**: Classification of movements based on Laban effort qualities (Time and Space)
- **Semi-Supervised Learning**: Leverages both labeled and unlabeled data for more robust model training
- **Dance Visualization**: Tools for visualizing dance sequences as animations

## Getting Started

### Prerequisites

```
python 3.7+
torch
transformers
dash (for labeling tool)
dash-bootstrap-components (for labeling tool)
numpy
scikit-learn
wandb (for experiment tracking)
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd human-ai-choreo

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Labeling

To annotate dance sequences with Laban effort qualities:

```bash
cd label_generation/skelabel
python3 app.py
```

The PirouNet Labeler will be available at http://localhost:8060

### Training Models

To train the semi-supervised model that maps between text descriptions and dance movements:

```bash
python3 train_semi_supervised.py
```

This will train both text and pose encoders using contrastive learning, saving the models to the project root directory.

### Generating Dance from Text

To generate dance sequences from textual descriptions:

```bash
python3 test_generation.py
```

This script demonstrates how to use the trained models to find dance sequences that match textual descriptions like "Quick and Direct movement" or "Sustained and Indirect movement".

### Visualizing Dance

To visualize dance sequences:

```bash
python3 visualize_motion.ipynb
```

This creates animations of the dance sequences using matplotlib.

## Model Architecture

The project uses a dual-encoder architecture:

- **Text Encoder**: Fine-tuned BERT model that converts text descriptions into embeddings
- **Pose Encoder**: LSTM-based model that converts dance sequences into embeddings
- **Projector Heads**: MLP layers that project embeddings into a shared space for contrastive learning

The models are trained using contrastive learning to align the text and pose embeddings in a shared latent space.

## Laban Movement Analysis

The project focuses on two Laban effort qualities:

- **Time**: Ranges from Sustained (1) to Sudden/Quick (3)
- **Space**: Ranges from Indirect (1) to Direct (3)

These qualities are combined to create descriptive labels like "Sustained and Direct" or "Quick and Indirect".

## Contributing

To extend or modify this project:

1. For the labeling tool:
   - Add new controls in Controls.py
   - Define their behavior in Callbacks.py
   - Add any required calculations in Calculations.py
   - Connect everything in app.py

2. For the models:
   - Modify model architectures in models/model.py
   - Adjust training procedures in train_semi_supervised.py

# Reasoning behind architecture decision
1. For finetuning I am using a classifier instead of text output because there are only 9 possible output combinations for each of the sequences in this case.
2. For pretraining I tried using text-sequence pair and sequence-augmented-sequence pair both but only using  sequence-augmented-sequence  didn't yiedl good results
3. I used the objective similar to the simclr/clip paper because of the success of the clip paper and the simclr paper in image text pair/image image pair based pretraining.
4. For pretraining I trained on seq-seq pairs for 5 epochs and then seq-aug-seq pairs and seq-text pairs for another 5 epochs


Example of results in hold out set: (I'm submitting this because I started late (but definitely could improve upon this (I used text-seq pair only for the pretraining - the loss for validation was better in that case))) - this could maybe a clubbing dance 🤣 (I could have also considered a frame as an image and used a pretrained clip model embedding for training (might have improved the results as it would then have some better representation))

https://github.com/user-attachments/assets/bccd9ca0-9f86-42c6-98c4-cf5112e9ff76

For seq to text task:
I have added example in the code: Sustained and Indirect movement is the output that I am getting


## Acknowledgments
- Thanks to the Laban Movement Analysis project for providing the dataset and analysis tools
- Special thanks to the Hugging Face Transformers library for their powerful NLP capabilities

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

