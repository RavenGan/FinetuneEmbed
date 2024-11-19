import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle

def split_data(genes, labels, gene_descriptions, save_dir,
               test_size=0.1, random_state=42):
    # Split into train and test sets
    genes_train, genes_test, labels_train, labels_test = train_test_split(genes, labels, 
                                                                          test_size=test_size, 
                                                                          stratify=labels, 
                                                                          random_state=random_state)
    desc_train = [gene_descriptions[gene] for gene in genes_train]
    desc_test = [gene_descriptions[gene] for gene in genes_test]

    # Save the data
    train_to_save = {'genes':genes_train, 'desc':desc_train, 'labels':labels_train}
    val_to_save = {'genes':genes_test, 'desc':desc_test, 'labels':labels_test}

    train_dir = save_dir + "/train_data_" + str(random_state) + ".pkl" 
    val_dir = save_dir + "/test_data_" + str(random_state) + ".pkl"
    # Save as a pickle file
    with open(train_dir, "wb") as f:
        pickle.dump(train_to_save, f)
    with open(val_dir, "wb") as f:
        pickle.dump(val_to_save, f)

def load_data(train_dict, test_dict, embed_dict):
    # Get gene names and the corresponding labels
    train_genes = train_dict['genes']
    train_labels = train_dict['labels']

    test_genes = test_dict['genes']
    test_labels = test_dict['labels']

    # Get the intersected genes
    overlap_train_gene = list(set(train_genes) & set(embed_dict.keys()))
    overlap_test_gene = list(set(test_genes) & set(embed_dict.keys()))

    train_indices = [train_genes.index(x) for x in overlap_train_gene]
    overlap_train_labels = [train_labels[i] for i in train_indices]

    test_indices = [test_genes.index(x) for x in overlap_test_gene]
    overlap_test_labels = [test_labels[i] for i in test_indices]

    X_train = [embed_dict[x] for x in overlap_train_gene if x in embed_dict]
    X_train = np.array(X_train)
    y_train = np.array(overlap_train_labels)

    X_test = [embed_dict[x] for x in overlap_test_gene if x in embed_dict]
    X_test = np.array(X_test)
    y_test = np.array(overlap_test_labels)

    return X_train, y_train, X_test, y_test


def LogisticReg_cv(X_train, y_train, folds=5): # default 5-fold
    # Initialize the logistic regression model
    log_reg = LogisticRegression()

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)

    # Variables to track the best model and its performance
    best_model = None
    best_val_auc = 0

    val_auc_scores = []

    # Loop through each fold for cross-validation
    for train_idx, val_idx in cv.split(X_train, y_train):
        # Split data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Train the logistic regression model on the training fold
        log_reg.fit(X_train_fold, y_train_fold)
        
        # Predict probabilities on the validation fold
        y_val_pred_proba = log_reg.predict_proba(X_val_fold)[:, 1]
        
        # Calculate AUC for the validation fold
        val_auc = roc_auc_score(y_val_fold, y_val_pred_proba)
        val_auc_scores.append(val_auc)

        # Update the best model if the current fold's AUC is higher
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = log_reg  # Store the model trained on this fold
    
    mean_val_auc = np.mean(val_auc_scores)
    std_val_auc = np.std(val_auc_scores)
    print(f"Validation AUC: Mean = {mean_val_auc:.4f}, Standard Deviation = {std_val_auc:.4f}")

    return best_model


def RandomForest_cv(X_train, y_train, folds=5): # default 5-fold
    # Initialize the random forest model
    random_forest = RandomForestClassifier()

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)

    # Variables to track the best model and its performance
    best_model = None
    best_val_auc = 0

    val_auc_scores = []

    # Loop through each fold for cross-validation
    for train_idx, val_idx in cv.split(X_train, y_train):
        # Split data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Train the random forest model on the training fold
        random_forest.fit(X_train_fold, y_train_fold)
        
        # Predict probabilities on the validation fold
        y_val_pred_proba = random_forest.predict_proba(X_val_fold)[:, 1]
        
        # Calculate AUC for the validation fold
        val_auc = roc_auc_score(y_val_fold, y_val_pred_proba)
        val_auc_scores.append(val_auc)

        # Update the best model if the current fold's AUC is higher
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = random_forest  # Store the model trained on this fold
    
    mean_val_auc = np.mean(val_auc_scores)
    std_val_auc = np.std(val_auc_scores)
    print(f"Validation AUC: Mean = {mean_val_auc:.4f}, Standard Deviation = {std_val_auc:.4f}")

    return best_model
