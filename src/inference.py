import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pickle
from model.transformer import GPTModel
from model.tokenizer import Tokenizer
import time

def generate_text(model, tokenizer, start_text, max_length=100, temperature=0.7, print_speed=0.05):
    model.eval()
    tokens = tokenizer.encode(start_text)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    generated_text = start_text
    print(start_text, end='', flush=True)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, None)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
            
            if next_token == tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            next_word = tokenizer.decode([next_token])
            if next_word != '<UNK>':
                if not next_word.startswith((' ', ',', '.', '!', '?')):
                    generated_text += ' '
                    print(' ', end='', flush=True)
                generated_text += next_word
                print(next_word, end='', flush=True)
                time.sleep(print_speed)
    
    print()  # New line after generation
    return generated_text

def main():
    tokenizer_path = 'trained/tokenizer.pkl'
    model_path = 'trained/model.pth'

    if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        print("Tokenizer or model not found. Please train the model first.")
        return

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    old_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]

    model = GPTModel(vocab_size=tokenizer.vocab_size, d_model=256, nhead=8, d_hid=512, nlayers=4)
    
    if old_vocab_size != tokenizer.vocab_size:
        print(f"Vocabulary size changed from {old_vocab_size} to {tokenizer.vocab_size}. Resizing model...")
        model.resize_token_embeddings(tokenizer.vocab_size)
    
    # Load state dict after resizing
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    while True:
        start_text = input("What do you want to ask? (Type 'exit' to quit): ")
        
        if start_text.lower() == 'exit':
            break
        
        generate_text(model, tokenizer, start_text, print_speed=0.10)
        print("\n")

if __name__ == "__main__":
    main()
