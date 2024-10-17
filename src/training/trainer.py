import torch
import torch.nn as nn
import time

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, data, batch_size, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            data = data.to(self.device)
            
            for i in range(0, len(data) - batch_size):
                input_seq = data[i:i+batch_size]
                target_seq = data[i+1:i+1+batch_size]

                self.optimizer.zero_grad()
                output = self.model(input_seq.unsqueeze(0), None)
                
                loss = self.criterion(output.squeeze(0), target_seq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

    def _get_batch(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data.to(self.device), target.to(self.device)
