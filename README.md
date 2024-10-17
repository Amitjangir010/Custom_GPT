# My GPT Model

## Project Description

This project implements a custom GPT (Generative Pre-trained Transformer) model for text generation. It provides functionality for training on custom data, fine-tuning, and generating text based on the trained model.

**Note**: This project is created primarily for testing purposes, to explore how a GPT-like model can be trained from scratch. The idea is to understand the inner workings of a GPT model, how it can be trained on custom data, and how we can fine-tune it according to our needs. However, it's important to mention that the results might not be great even after training with several epochs or datasets because GPT models require a massive amount of high-quality data to perform well. This is a fun project meant to experiment with training GPT models and selecting layers, heads, and other hyperparameters based on different use cases.

## Features

- Custom tokenizer for efficient text processing
- GPT model implementation using PyTorch
- Training script for fine-tuning on custom datasets
- Inference script for text generation
- Incremental model updates with new data
- Utility functions for data management and preprocessing

## Project Structure

```
My_gpt/
│
├── src/
│   ├── model/
│   │   ├── transformer.py      # GPT model implementation
│   │   └── tokenizer.py        # Custom tokenizer for text
│   ├── training/
│   │   └── trainer.py          # Training script for fine-tuning the model
│   ├── utils/
│   │   └── data_utils.py       # Utility functions for data processing
│   ├── main.py                 # Main script for training and model updates
│   └── inference.py            # Script for text generation
│
├── data/
│   ├── daily_thoughts.txt      # Custom dataset for training
│   └── all_thoughts.txt        # Full dataset for model fine-tuning
│
├── trained/
│   ├── model.pth               # Saved model checkpoint
│   └── tokenizer.pkl           # Saved tokenizer
│
└── README.md
```

## Setup and Installation

Follow the steps below to set up and run the project.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/My_gpt.git
   cd My_gpt
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install torch
   ```

## Usage

### Training

To train or fine-tune the GPT model:

1. Add your text data to `data/daily_thoughts.txt`.
2. Run the training script:

   ```bash
   python src/main.py
   ```

### Inference

To generate text using the trained model:

1. Run the inference script:

   ```bash
   python src/inference.py
   ```

2. Enter a prompt when asked, and the model will generate text based on your input.

## Components

- `model/transformer.py`: Implements the GPT architecture using PyTorch.
- `model/tokenizer.py`: Custom tokenizer for processing text input.
- `training/trainer.py`: Script for training the GPT model.
- `utils/data_utils.py`: Utility functions for managing and preprocessing data.
- `main.py`: Main script to trigger training or model updates.
- `inference.py`: Script for generating text based on the trained model.

## Notes

- The model’s vocabulary dynamically expands with new data during training.
- Incremental training allows for continuous improvement of the model with new datasets.
- Utility functions in `data_utils.py` streamline data preparation and management.
- Users can customize hyperparameters like layers, heads, etc., according to their use case to experiment with different configurations.

## Future Improvements

- Implement advanced text cleaning techniques.
- Add support for multiple GPT architectures and hyperparameter tuning.
- Enhance tokenization for special characters and multilingual text.
- Improve data management utilities for larger datasets.

## Contributing

Contributions are welcome! Please feel free to submit a pull request with improvements or new features.