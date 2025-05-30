import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
import pandas as pd
import pickle
import csv
import os

def TrainEvalTest_Name_split(genes, labels, save_dir,
                            test_size=0.1, eval_size=0.1, random_state=42):
    # Split into 90% train+val, 10% test
    genes_train_eval, genes_test, labels_train_eval, labels_test = train_test_split(genes, labels, 
                                                                          test_size=test_size, 
                                                                          stratify=labels, 
                                                                          random_state=random_state)
    # Split the 90% train+val into 80% train, 100% val
    genes_train, genes_eval, labels_train, labels_eval = train_test_split(genes_train_eval, labels_train_eval, 
                                                                          test_size=eval_size/(1-test_size), 
                                                                          stratify=labels_train_eval, 
                                                                          random_state=random_state)
    
    name_train = [gene for gene in genes_train]
    name_eva = [gene for gene in genes_eval]
    name_test = [gene for gene in genes_test]

    # Save the data
    train_to_save = {'genes':genes_train, 'desc':name_train, 'labels':labels_train}
    eval_to_save = {'genes':genes_eval, 'desc':name_eva, 'labels':labels_eval}
    val_to_save = {'genes':genes_test, 'desc':name_test, 'labels':labels_test}

    train_dir = save_dir + "/train_data_" + str(random_state) + ".pkl" 
    eval_dir = save_dir + "/eval_data_" + str(random_state) + ".pkl"
    val_dir = save_dir + "/test_data_" + str(random_state) + ".pkl"

    # Save as a pickle file
    with open(train_dir, "wb") as f:
        pickle.dump(train_to_save, f)
    with open(eval_dir, "wb") as f:
        pickle.dump(eval_to_save, f)
    with open(val_dir, "wb") as f:
        pickle.dump(val_to_save, f)



def TrainEvalTest_split(genes, labels, gene_descriptions, save_dir, 
                        test_size=0.1, eval_size=0.1, random_state=42):
    # Split into 90% train+val, 10% test
    genes_train_eval, genes_test, labels_train_eval, labels_test = train_test_split(genes, labels, 
                                                                          test_size=test_size, 
                                                                          stratify=labels, 
                                                                          random_state=random_state)
    # Split the 90% train+val into 80% train, 100% val
    genes_train, genes_eval, labels_train, labels_eval = train_test_split(genes_train_eval, labels_train_eval, 
                                                                          test_size=eval_size/(1-test_size), 
                                                                          stratify=labels_train_eval, 
                                                                          random_state=random_state)
    
    desc_train = [gene_descriptions[gene] for gene in genes_train]
    desc_eva = [gene_descriptions[gene] for gene in genes_eval]
    desc_test = [gene_descriptions[gene] for gene in genes_test]

    # Save the data
    train_to_save = {'genes':genes_train, 'desc':desc_train, 'labels':labels_train}
    eval_to_save = {'genes':genes_eval, 'desc':desc_eva, 'labels':labels_eval}
    val_to_save = {'genes':genes_test, 'desc':desc_test, 'labels':labels_test}

    train_dir = save_dir + "/train_data_" + str(random_state) + ".pkl" 
    eval_dir = save_dir + "/eval_data_" + str(random_state) + ".pkl"
    val_dir = save_dir + "/test_data_" + str(random_state) + ".pkl"

    # Save as a pickle file
    with open(train_dir, "wb") as f:
        pickle.dump(train_to_save, f)
    with open(eval_dir, "wb") as f:
        pickle.dump(eval_to_save, f)
    with open(val_dir, "wb") as f:
        pickle.dump(val_to_save, f)



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

def load_data(train_dict, test_dict, embed_dict, do_pca=False):
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

    if do_pca:
        pca = PCA(n_components=40)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

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
    print(f"LR Validation AUC: Mean = {mean_val_auc:.4f}, Standard Deviation = {std_val_auc:.4f}")

    return best_model, mean_val_auc, std_val_auc


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
    print(f"RF Validation AUC: Mean = {mean_val_auc:.4f}, Standard Deviation = {std_val_auc:.4f}")

    return best_model, mean_val_auc, std_val_auc


def get_LR_RF_res(train_dict, test_dict, embed_dict):
    # Load the data
    X_train, y_train, X_test, y_test = load_data(train_dict, test_dict, embed_dict)

    ### Logistic regression results
    best_model, LR_mean_val_auc, LR_std_val_auc = LogisticReg_cv(X_train, y_train, folds=5)
    # Train the best model on the full training data
    best_model.fit(X_train, y_train)
    # Evaluate the best model on the test set
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    LR_test_auc = roc_auc_score(y_test, y_test_pred_proba)

    ### Random forest results
    best_model, RF_mean_val_auc, RF_std_val_auc = RandomForest_cv(X_train, y_train, folds=5)
    # Train the best model on the full training data
    best_model.fit(X_train, y_train)
    # Evaluate the best model on the test set
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    RF_test_auc = roc_auc_score(y_test, y_test_pred_proba)

    return LR_mean_val_auc, LR_test_auc, RF_mean_val_auc, RF_test_auc

def multiple_run(data_dir, save_csv_dir, random_states, embed_dict):
    LR_val_res = []
    RF_val_res = []
    LR_test_res = []
    RF_test_res = []

    for random_state in random_states:
        train_dir = data_dir + "/TrainTestData/train_data_" + str(random_state) + ".pkl"
        test_dir = data_dir + "/TrainTestData/test_data_" + str(random_state) + ".pkl"
        # prepare the input data
        with open(train_dir, "rb") as f:
            train_data = pickle.load(f)
        with open(test_dir, "rb") as f:
            test_data = pickle.load(f)

        LR_mean_val_auc, LR_test_auc, RF_mean_val_auc, RF_test_auc = get_LR_RF_res(train_data, 
                                                                                test_data, 
                                                                                embed_dict)
        LR_val_res.append(f"{LR_mean_val_auc:.4f}")
        RF_val_res.append(f"{RF_mean_val_auc:.4f}")
        LR_test_res.append(f"{LR_test_auc:.4f}")
        RF_test_res.append(f"{RF_test_auc:.4f}")

    rows = zip(LR_val_res, RF_val_res, LR_test_res, RF_test_res)

    # Write to a CSV file
    with open(save_csv_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optional: Write a header row
        writer.writerow(['LR_val', 'RF_val', 'LR_test', 'RF_test'])
        # Write the rows
        writer.writerows(rows)


def load_data_TrainTest(train_dict, eval_dict, test_dict, embed_dict, do_pca, n_PCs, do_truncation):
    # Get gene names and the corresponding labels
    train_genes = train_dict['genes']
    train_labels = train_dict['labels']

    eval_genes = eval_dict['genes']
    eval_labels = eval_dict['labels']

    test_genes = test_dict['genes']
    test_labels = test_dict['labels']

    # Combine the training and validation data
    train_genes = train_genes + eval_genes
    train_labels = train_labels + eval_labels

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

    # If PCA is needed, apply PCA here
    if do_pca:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=n_PCs) # number of components to keep
        X_train_pca = pca.fit_transform(X_train_scaled)

        X_test_pca = pca.transform(X_test_scaled)

        return X_train_pca, y_train, X_test_pca, y_test
    
    elif do_truncation:
        # If truncation is needed, truncate the embeddings to 384
        X_train_truncated = X_train[:, :384]
        X_test_truncated = X_test[:, :384]

        return X_train_truncated, y_train, X_test_truncated, y_test
    else:
        # If PCA is not needed, return the original data
        return X_train, y_train, X_test, y_test


def LogisticReg_TrainTest(X_train, y_train, X_test, y_test, save_path):
    # Initialize the logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Get predicted probabilities and classes
    y_test_pred = log_reg.predict(X_test)
    y_test_proba = log_reg.predict_proba(X_test)

    # Determine if binary or multi-class
    classes = np.unique(y_train)
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    # Metrics
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')

    # ROC-AUC
    try:
        if is_multiclass:
            test_auc = roc_auc_score(y_test, y_test_proba, average='macro', multi_class='ovr')
            y_test_bin = label_binarize(y_test, classes=classes)  # For plotting ROC curves
            # y_test_bin = label_binarize(y_test, classes=classes)
            # test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
        else:
            y_test_bin = y_test  # keep as is for binary
            test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
        test_auc = np.nan

    # Save ROC-related data for later use
    try:
        roc_data = {
            'y_test_bin': y_test_bin,
            'y_test_proba': y_test_proba,
            'classes': classes
        }
        save_path = save_path + "/LR_roc_data.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(roc_data, f)
        print(f"Saved ROC data to {save_path}")
    except Exception as e:
        print("Could not save ROC data:", e)

    res_dict = {
        "AUC": test_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    # keep only 3 decimal places
    res_dict = {k: round(v, 3) for k, v in res_dict.items()}

    return res_dict

def LogisticReg_TrainTest_CV(X_train, y_train, X_test, y_test, save_path):
    # Determine class info
    classes = np.unique(y_train)
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    # Define the logistic regression model
    model = LogisticRegression(solver='liblinear')  # use liblinear for small datasets and L1 penalty
    # Define a range of hyperparameters for tuning
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    
    # Initialize GridSearchCV with cross-validation (CV) on the training data
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', #'accuracy', 
                               verbose=1)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # report the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # load the best model and evaluate on the test data
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)

    # Metrics
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')

    # ROC-AUC score and save binarized data for plotting later
    try:
        if is_multiclass:
            test_auc = roc_auc_score(y_test, y_test_proba, average='macro', multi_class='ovr')
            y_test_bin = label_binarize(y_test, classes=classes)  # For plotting ROC curves
            # y_test_bin = label_binarize(y_test, classes=classes)
            # test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
        else:
            y_test_bin = y_test  # keep as is for binary
            test_auc = roc_auc_score(y_test, y_test_proba[:, 1])

        # Save ROC data for future plotting
        roc_data = {
            'y_test_bin': y_test_bin,
            'y_test_proba': y_test_proba,
            'classes': classes
        }
        save_path = save_path + "/LR_CV_roc_data.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(roc_data, f)
        print(f"ROC data saved to: {save_path}")

    except Exception as e:
        print("ROC-AUC skipped due to error:", e)
        test_auc = np.nan
    
    res_dict = {
        "AUC": test_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    # keep only 3 decimal places
    res_dict = {k: round(v, 3) for k, v in res_dict.items()}

    return res_dict

def RandomForest_TrainTest(X_train, y_train, X_test, y_test, save_path):
    # Initialize the random forest model
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    # Predict probabilities and classes
    y_test_pred = random_forest.predict(X_test)
    y_test_proba = random_forest.predict_proba(X_test)

    # Determine binary vs. multi-class
    classes = np.unique(y_train)
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    # Compute metrics
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')

    # Compute ROC-AUC and prepare ROC data
    try:
        if is_multiclass:
            test_auc = roc_auc_score(y_test, y_test_proba, average='macro', multi_class='ovr')
            y_test_bin = label_binarize(y_test, classes=classes)  # For plotting ROC curves
            # y_test_bin = label_binarize(y_test, classes=classes)
            # test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
        else:
            y_test_bin = y_test  # binary case
            test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
        y_test_bin = None
        test_auc = np.nan
    
    # Save ROC-related data
    try:
        roc_data = {
            'y_test_bin': y_test_bin,
            'y_test_proba': y_test_proba,
            'classes': classes
        }
        save_path = save_path + "/RF_roc_data.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(roc_data, f)
        print(f"Saved ROC data to {save_path}")
    except Exception as e:
        print("Could not save ROC data:", e)

    res_dict = {
        "AUC": test_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    # keep only 3 decimal places
    res_dict = {k: round(v, 3) for k, v in res_dict.items()}

    return res_dict

def RandomForest_TrainTest_CV(X_train, y_train, X_test, y_test, save_path):
    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier()
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [25, 50, 100, 200, 400],        # Number of trees in the forest
        'max_depth': [None, 10, 20, 30]            # Maximum depth of each tree
    }
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='roc_auc', # 'accuracy', 
                               verbose=1, 
                               n_jobs=-1 # Use all available cores
                               )
    # Fit the model with the training data
    grid_search.fit(X_train, y_train)
    # report the best parameters
    print("Best hyperparameters found: ", grid_search.best_params_)

    # Get the best model from the grid search
    best_rf_model = grid_search.best_estimator_
    y_test_pred = best_rf_model.predict(X_test)
    y_test_proba = best_rf_model.predict_proba(X_test)

    # Determine binary vs. multi-class
    classes = np.unique(y_train)
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    # Compute metrics
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')

    # Compute ROC-AUC and prepare ROC data
    try:
        if is_multiclass:
            test_auc = roc_auc_score(y_test, y_test_proba, average='macro', multi_class='ovr')
            y_test_bin = label_binarize(y_test, classes=classes)  # For plotting ROC curves
            # y_test_bin = label_binarize(y_test, classes=classes)
            # test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
        else:
            y_test_bin = y_test
            test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
        y_test_bin = None
        test_auc = np.nan
    
    # Save ROC-related data
    try:
        roc_data = {
            'y_test_bin': y_test_bin,
            'y_test_proba': y_test_proba,
            'classes': classes
        }
        save_path = save_path + "/RF_CV_roc_data.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(roc_data, f)
        print(f"Saved ROC data to {save_path}")
    except Exception as e:
        print("Could not save ROC data:", e)

    res_dict = {
        "AUC": test_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    # keep only 3 decimal places
    res_dict = {k: round(v, 3) for k, v in res_dict.items()}

    return res_dict

def get_LR_RF_res_TrainTest(train_dict, eval_dict, test_dict, embed_dict, 
                            do_pca, do_cv, ROC_save_path, n_PCs, do_truncation):
    # Load the data
    X_train, y_train, X_test, y_test = load_data_TrainTest(train_dict, eval_dict, test_dict, embed_dict, 
                                                           do_pca, n_PCs, do_truncation)

    if do_cv:
        LR_test_auc = LogisticReg_TrainTest_CV(X_train, y_train, X_test, y_test, ROC_save_path)
        RF_test_auc = RandomForest_TrainTest_CV(X_train, y_train, X_test, y_test, ROC_save_path)
    else:
        LR_test_auc = LogisticReg_TrainTest(X_train, y_train, X_test, y_test, ROC_save_path)
        RF_test_auc = RandomForest_TrainTest(X_train, y_train, X_test, y_test, ROC_save_path)


    return LR_test_auc, RF_test_auc


def multiple_run_TrainTest(data_dir, save_csv_dir, random_states, 
                           embed_dict, ROC_save_path, do_cv, do_pca, n_PCs, do_truncation):
    LR_test_res = []
    RF_test_res = []

    for random_state in random_states:
        train_dir = data_dir + "/train_data_" + str(random_state) + ".pkl"
        eval_dir = data_dir + "/eval_data_" + str(random_state) + ".pkl"
        test_dir = data_dir + "/test_data_" + str(random_state) + ".pkl"
        # prepare the input data
        with open(train_dir, "rb") as f:
            train_data = pickle.load(f)
        with open(eval_dir, "rb") as f:
            eval_data = pickle.load(f)
        with open(test_dir, "rb") as f:
            test_data = pickle.load(f)
        
        save_path = ROC_save_path + "random_state_" + str(random_state)

        LR_test_auc, RF_test_auc = get_LR_RF_res_TrainTest(train_data, eval_data, test_data, 
                                                           embed_dict, do_pca, do_cv, save_path, n_PCs,
                                                           do_truncation)
        LR_test_res.append(LR_test_auc)
        RF_test_res.append(RF_test_auc)

    LR_df = pd.DataFrame(LR_test_res)
    LR_df['model'] = "LR"
    RF_df = pd.DataFrame(RF_test_res)
    RF_df['model'] = "RF"

    # Combine both
    combined_df = pd.concat([LR_df, RF_df], ignore_index=True)

    # Save to CSV
    combined_df.to_csv(save_csv_dir, index=False)


