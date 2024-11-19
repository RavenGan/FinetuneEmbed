import torch
from torch.utils.data import Dataset
from transformers import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the tokenizer (squeeze the single dimension)
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        encoding["label"] = torch.tensor(label, dtype=torch.long)

        return encoding
    
# Define the custom trainer class using ReduceLROnPlateau scheduler
class CustomTrainer(Trainer):
    def __init__(self, *args, eval_metric="AUC", **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_metric = eval_metric
        self.reduce_lr_scheduler = None  # Initialize as None

    def train(self, *args, **kwargs):
        # Initialize the optimizer and standard scheduler
        output = super().train(*args, **kwargs)
        
        # Create ReduceLROnPlateau scheduler after the optimizer has been created
        self.reduce_lr_scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.8, patience=2, verbose=False)
        return output

    def evaluate(self, *args, **kwargs):
        # Evaluate and store the results
        eval_output = super().evaluate(*args, **kwargs)
        
        # Access the chosen evaluation metric and step the scheduler
        metric_value = eval_output[f"eval_{self.eval_metric}"]
        
        # Step the scheduler if it’s initialized
        if self.reduce_lr_scheduler:
            self.reduce_lr_scheduler.step(metric_value)
        
        return eval_output


# Define the compute_metrics function for AUC
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # Get probability of the positive class
    auc = roc_auc_score(labels, probs)
    return {"AUC": auc}

def one_fold_training(train_texts, train_labels, val_texts, val_labels, tokenizer,
                      output_dir):
    # Create PyTorch datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
