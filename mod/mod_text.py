import torch
from torch.utils.data import Dataset
from transformers import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback

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


def one_fold_training(train_texts, train_labels, 
                      val_texts, val_labels, 
                      tokenizer, output_dir, fold,
                      model_name):
    # Create PyTorch datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    # Define output directory for this fold
    output_dir_full = output_dir + f"/fold_{fold + 1}"
    os.makedirs(output_dir_full, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_full,
        evaluation_strategy="epoch",
        save_strategy="epoch", # Save checkpoints at the end of each epoch
        load_best_model_at_end=True, # Load the best model at the end of each fold
        save_total_limit=1, # Keep only the best model checkpoint
        learning_rate=1e-4, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,
        metric_for_best_model="AUC",
        greater_is_better=True
    )

    # Initialize the model and Trainer for this fold
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels=2)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        eval_metric="AUC",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train the model on this fold
    trainer.train()
    trainer.save_model(output_dir_full)

    # Evaluate on the validation set and save the best model's AUC
    val_results = trainer.evaluate()
    val_auc = val_results["eval_AUC"]
    print(f"Fold {fold + 1} Validation AUC: {val_auc}")

    return output_dir_full, val_auc

def multi_fold_training(train_texts_all, train_labels_all,
                        model_name, output_dir, n_splits=5):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)

    # Initialize a list to store AUC scores for each fold
    val_auc_scores = []
    output_dirs = []  # Track output directories for each fold

    for fold, (train_index, val_index) in enumerate(kf.split(train_texts_all, train_labels_all)):
        print(f"Fold {fold + 1}/{n_splits}")
        # Split data into training and validation for this fold
        train_texts, val_texts = [train_texts_all[i] for i in train_index], [train_texts_all[i] for i in val_index]
        train_labels, val_labels = [train_labels_all[i] for i in train_index], [train_labels_all[i] for i in val_index]

        output_dir_full, val_auc = one_fold_training(train_texts, train_labels, 
                                                val_texts, val_labels, 
                                                tokenizer, output_dir, fold, model_name)
        output_dirs.append(output_dir_full)
        val_auc_scores.append(val_auc)
    
    return output_dirs, val_auc_scores

def finetune_best_mod(full_train_dataset, best_model, output_dir):
    # Define training arguments for the final training phase
    final_training_args = TrainingArguments(
        output_dir=output_dir,       # Directory to save the final model
        evaluation_strategy="no",         # No evaluation during training
        save_strategy="no",            # Save the model at each epoch
        save_total_limit=1,               # Keep only the last checkpoint to save storage
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10000,              # Minimize logging output
        report_to="none"                  # Disable logging to external tools
    )
    # Initialize the Trainer with the full dataset and final training arguments
    trainer = Trainer(
        model=best_model,
        args=final_training_args,
        train_dataset=full_train_dataset
    )
    # Train the model on the full dataset
    trainer.train()
    return trainer

def pred_test(final_trainer, test_dataset):
    # Predict on the test dataset
    test_results = final_trainer.predict(test_dataset)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=1)[:, 1].numpy()
    test_auc = roc_auc_score(test_results.label_ids, test_probs)
    print(f"Test AUC: {test_auc}")
    return test_auc
