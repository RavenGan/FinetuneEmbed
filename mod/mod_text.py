import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

# Define custom classification model with SBERT embeddings
class SBERTClassifier(nn.Module):
    def __init__(self, sbert_model):
        super(SBERTClassifier, self).__init__()
        self.sbert_model = sbert_model
        self.classifier = nn.Sequential(
            nn.Linear(self.sbert_model.get_sentence_embedding_dimension(), 1),
            nn.Sigmoid()
        )

        # Unfreeze SBERT parameters to allow fine-tuning
        for param in self.sbert_model.parameters():
            param.requires_grad = True  # Do not allow SBERT layers to be trainable
    
    def forward(self, input_texts):
        # Encode the input texts using SBERT
        embeddings = self.sbert_model.encode(input_texts, convert_to_tensor=True)
        logits = self.classifier(embeddings)
        return logits


# # Example Dataset class
# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels):
#         self.texts = texts
#         self.labels = labels
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

# Custom Dataset class
class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, genes: List[str], labels: List[int], descriptions: List[str]):
        self.genes = genes
        self.labels = labels
        self.descriptions = descriptions

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        gene = self.genes[idx]
        label = self.labels[idx]
        description = self.descriptions[idx]
        return {"gene": gene, "label": torch.tensor(label), "description": description}

# Custom collate function for variable-length descriptions
def collate_fn(batch: List[Dict]):
    genes = [item['gene'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Tokenize or convert descriptions to fixed-size encoding as needed
    # For demonstration, assume each description is converted to a list of tokens
    descriptions = [torch.tensor([ord(char) for char in item['description']]) for item in batch]  # Example encoding
    descriptions_padded = pad_sequence(descriptions, batch_first=True, padding_value=0)
    
    return {"genes": genes, "labels": labels, "descriptions": descriptions_padded}



# Training function with AUC calculation on validation set
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', lr_patience=2):
    model.train()
    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=lr_patience, factor=0.5, verbose=True)
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

            # print(outputs.shape)
            # print(labels)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
        
        # Validate after each epoch
        auc = evaluate_model(model, val_loader, device)
        print(f"Validation AUC after Epoch {epoch + 1}: {auc:.4f}")
        # Step the scheduler based on validation AUC
        scheduler.step(auc)

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