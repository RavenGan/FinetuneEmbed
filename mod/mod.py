import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Define custom classification model with SBERT embeddings
class SBERTClassifier(nn.Module):
    def __init__(self, sbert_model):
        super(SBERTClassifier, self).__init__()
        self.sbert_model = sbert_model
        self.classifier = nn.Sequential(
            nn.Linear(self.sbert_model.get_sentence_embedding_dimension(), 128),
            nn.ReLU(),  # Activation function
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Sigmoid to output probabilities between 0 and 1
        )
    
    def forward(self, input_texts):
        # Encode the input texts using SBERT
        embeddings = self.sbert_model.encode(input_texts, convert_to_tensor=True)
        logits = self.classifier(embeddings)
        return logits


# Example Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


import numpy as np
# Training function with AUC calculation on validation set
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for texts, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            texts = [text for text in texts]
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(texts).squeeze()  # Remove extra dimension

            if outputs.ndim == 0:  # If outputs is scalar, reshape it to [1]
                outputs = outputs.unsqueeze(0)
            if labels.ndim == 0:  # If labels is scalar, reshape it to [1]
                labels = labels.unsqueeze(0)

            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
        
        # Validate after each epoch
        auc = evaluate_model(model, val_loader, device)
        print(f"Validation AUC after Epoch {epoch + 1}: {auc:.4f}")

# Evaluation function for AUC
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = [text for text in texts]
            labels = labels.to(device)
            
            # Get predicted probabilities
            outputs = model(texts).squeeze()  # Remove extra dimension
            all_labels.extend(labels.cpu().detach().numpy())

            outputs = outputs.cpu().detach().numpy()
            if outputs.ndim == 0:
                outputs = outputs.reshape(1)

            all_probs.extend(outputs)
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs)
    model.train()  # Set model back to training mode
    return auc

# Test function to calculate AUC on the test set
def test_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = [text for text in texts]
            labels = labels.to(device)
            
            outputs = model(texts).squeeze()
            all_labels.extend(labels.cpu().detach().numpy())

            outputs = outputs.cpu().detach().numpy()
            if outputs.ndim == 0:
                outputs = outputs.reshape(1)
            all_probs.extend(outputs)
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Test AUC: {auc:.4f}")