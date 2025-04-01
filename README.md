# Human-AI Choreography Project

## Overview
The Human-AI Choreography project is designed to explore the intersection of dance and artificial intelligence. It includes tools for annotating dance motion capture sequences, training models, and analyzing dance movements using Laban Movement Analysis.

## Project Structure
The project is organized into several components:

- **label_generation/skelabel**: Contains the PirouNet Labeler, a Dash-based web application for annotating dance motion capture sequences.
- **utils**: Includes utility functions and configurations for data processing and model training.
- **models**: Contains model definitions and training scripts.
- **data**: Stores motion capture data and animations.
- **training_dataset**: Contains labeled datasets for training.

## Getting Started

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd human-ai-choreo
pip install -r requirements.txt
```

### Running the Application
```bash
cd label_generation/skelabel
python app.py
```
The application will be available at http://localhost:8060

### Running Training Scripts
To train the model, use the following command:
```bash
python train.py
```

### Environment Configuration
Ensure that your environment has Python 3.7 or higher and all dependencies listed in `requirements.txt` are installed.

### Troubleshooting
- If you encounter issues with missing dependencies, ensure that you have activated your virtual environment and installed all required packages.
- For GPU support, ensure that CUDA is installed and configured correctly.

## Contributing
To extend or modify this application:
1. Add new controls in Controls.py
2. Define their behavior in Callbacks.py
3. Add any required calculations in Calculations.py
4. Connect everything in app.py

## License
This project is licensed under the MIT License. See the LICENSE file for more details.