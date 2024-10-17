import torch

def load_data(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer.fit(text)
    encoded_text = torch.tensor(tokenizer.encode(text))
    return encoded_text

def save_thoughts(file_path, thoughts):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(thoughts + '\n')

def merge_thoughts(daily_file, all_file):
    with open(daily_file, 'r', encoding='utf-8') as daily, open(all_file, 'a', encoding='utf-8') as all:
        all.write(daily.read())
    
    # Clear the daily thoughts file
    open(daily_file, 'w').close()
