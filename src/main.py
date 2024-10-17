import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pickle
from training.trainer import Trainer
from model.transformer import GPTModel
from model.tokenizer import Tokenizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Create 'trained' folder if it doesn't exist
    os.makedirs('trained', exist_ok=True)

    tokenizer_path = 'trained/tokenizer.pkl'
    model_path = 'trained/model.pth'

    # Load or create tokenizer
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Loaded previous tokenizer.")
    else:
        tokenizer = Tokenizer()
        print("Created new tokenizer.")

    # After updating the tokenizer
    tokenizer.eos_token_id = 2  # Ensure this is set

    # Load daily thoughts
    with open('data/daily_thoughts.txt', 'r', encoding='utf-8') as f:
        daily_text = f.read()

    if not daily_text.strip():
        print("No new data in daily_thoughts.txt. Skipping training.")
        return

    # Update tokenizer with new data
    tokenizer.fit(daily_text)

    # Save updated tokenizer
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Encode daily text
    encoded_text = torch.tensor(tokenizer.encode(daily_text), dtype=torch.long)

    # Initialize model
    model = GPTModel(vocab_size=tokenizer.vocab_size, d_model=256, nhead=8, d_hid=512, nlayers=4)
    
    # Print model parameters
    num_params = count_parameters(model)
    print(f"Number of model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Set up optimizer, criterion, and device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize trainer
    trainer = Trainer(model, optimizer, criterion, device)

    # Load previous model if exists
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path)
            old_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
            
            if old_vocab_size != tokenizer.vocab_size:
                print(f"Vocabulary size changed from {old_vocab_size} to {tokenizer.vocab_size}. Resizing model...")
                model.resize_token_embeddings(tokenizer.vocab_size)
            
            # Load state dict after resizing
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded and updated previous model.")
        except KeyError:
            print("Previous model structure is incompatible. Starting with a fresh model.")
    else:
        print("No previous model found. Starting from scratch.")

    # Fine-tune model on daily thoughts
    trainer.train(encoded_text, batch_size=32, num_epochs=10)

    # Save model
    trainer.save_model(model_path)

    # Append daily thoughts to all_thoughts
    with open('data/all_thoughts.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + daily_text)

    # Clear daily_thoughts file
    open('data/daily_thoughts.txt', 'w').close()

    print("Training complete. Model, tokenizer, and data updated.")

if __name__ == "__main__":
    main()
