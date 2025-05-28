import torch
from torch.utils.data import Dataset
from transformers import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize, LabelEncoder
import numpy as np
import os
import pickle
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback, AutoConfig


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
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # Get probability of the positive class
#     auc = roc_auc_score(labels, probs)
#     return {"AUC": auc}

def compute_metrics(eval_pred):
    # Unpack logits and labels
    logits, labels = eval_pred

    # Check if logits is a tuple and handle appropriately
    if isinstance(logits, tuple):
        logits = logits[0]  # Extract the actual logits if they're in a tuple

    # Convert logits and labels to tensors if necessary
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()

    # Number of classes
    num_classes = probs.shape[1]

    try:
        if num_classes == 2:
            # Binary classification – use probs[:, 1] as positive class probability
            auc = roc_auc_score(labels_np, probs[:, 1])
        else:
            # Multi-class classification – binarize labels for OVR AUC
            labels_bin = label_binarize(labels_np, classes=np.arange(num_classes))
            auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
        auc = np.nan

    return {"AUC": auc}

def one_fold_training(train_texts, train_labels, 
                      val_texts, val_labels, 
                      tokenizer, output_dir, fold,
                      model_name, args):
    
    # Label encoding: convert string labels to integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)

    
    # Create PyTorch datasets
    train_dataset = TextDataset(train_texts, train_labels_encoded, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels_encoded, tokenizer)

    # Define output directory for this fold
    output_dir_full = output_dir + f"/fold_{fold + 1}"
    os.makedirs(output_dir_full, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_full,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy, # Save checkpoints at the end of each epoch
        load_best_model_at_end=True, # Load the best model at the end of each fold
        save_total_limit=1, # Keep only the best model checkpoint
        learning_rate=args.learning_rate, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        metric_for_best_model="AUC",
        greater_is_better=True
    )

    # Initialize the model and Trainer for this fold
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels=args.num_classes)
    
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

    return output_dir_full, val_auc, label_encoder

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

def finetune_best_mod(full_train_dataset, best_model, output_dir, args):
    # Define training arguments for the final training phase
    final_training_args = TrainingArguments(
        output_dir=output_dir,       # Directory to save the final model
        evaluation_strategy=args.final_evaluation_strategy,         # No evaluation during training
        save_strategy=args.final_save_strategy,            # Save the model at each epoch
        save_total_limit=1,               # Keep only the last checkpoint to save storage
        learning_rate=args.final_learning_rate,
        per_device_train_batch_size=8,
        num_train_epochs=args.final_num_train_epochs,
        max_grad_norm=args.final_max_grad_norm,
        weight_decay=args.final_weight_decay,
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


def pred_test(final_trainer, test_dataset, save_path, label_encoder=None):
    # Predict on test dataset
    test_results = final_trainer.predict(test_dataset)
    logits = test_results.predictions
    labels = test_results.label_ids

    # Extract logits if needed
    if isinstance(logits, tuple):
        logits = logits[0]

    # Convert to tensors
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    # Debug: shape
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    preds = np.argmax(probs, axis=1)

    # Decode string labels if encoder is provided
    if label_encoder is not None:
        # Make sure predictions are also interpretable (optional)
        decoded_preds = label_encoder.inverse_transform(preds)
        decoded_labels = label_encoder.inverse_transform(labels_np)
        print(f"Decoded predictions (sample): {decoded_preds[:5]}")
        print(f"Decoded labels (sample): {decoded_labels[:5]}")
        
        # Use label-encoded ints for computing metrics (still needed)
        labels_np = label_encoder.transform(decoded_labels)

    # Determine number of classes
    num_classes = probs.shape[1]

    try:
        if num_classes == 2:
            auc = roc_auc_score(labels_np, probs[:, 1])
            fpr, tpr, _ = roc_curve(labels_np, probs[:, 1])
            roc_data = {"fpr": {1: fpr}, "tpr": {1: tpr}, "labels": labels_np}
        else:
            labels_bin = label_binarize(labels_np, classes=np.arange(num_classes))
            auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")

            fpr, tpr = {}, {}
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_data = {"fpr": fpr, "tpr": tpr, "labels": labels_np}
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
        auc = np.nan
        roc_data = {}

    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average="macro")

    print(f"Test AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # Save ROC curve data
    with open(save_path, "wb") as f:
        pickle.dump(roc_data, f)

    return {
        "AUC": round(auc, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3)
    }