import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import csv

def TrainEvalTest_split(genes, labels, gene_descriptions, save_dir, 
                        test_size=0.2, eval_size=0.2, random_state=42):
    # Split into 80% train+val, 20% test
    genes_train_eval, genes_test, labels_train_eval, labels_test = train_test_split(genes, labels, 
                                                                          test_size=test_size, 
                                                                          stratify=labels, 
                                                                          random_state=random_state)
    # Split the 80% train+val into 60% train, 20% val
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


def load_data_TrainTest(train_dict, eval_dict, test_dict, embed_dict):
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
    return X_train, y_train, X_test, y_test


def LogisticReg_TrainTest(X_train, y_train, X_test, y_test):
    # Initialize the logistic regression model
    log_reg = LogisticRegression()

    log_reg.fit(X_train, y_train)
    y_test_pred_proba = log_reg.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    return test_auc

def RandomForest_TrainTest(X_train, y_train, X_test, y_test):
    # Initialize the random forest model
    random_forest = RandomForestClassifier()

    random_forest.fit(X_train, y_train)
    y_test_pred_proba = random_forest.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    return test_auc

def get_LR_RF_res_TrainTest(train_dict, eval_dict, test_dict, embed_dict):
    # Load the data
    X_train, y_train, X_test, y_test = load_data_TrainTest(train_dict, eval_dict, test_dict, embed_dict)

    ### Logistic regression results
    LR_test_auc = LogisticReg_TrainTest(X_train, y_train, X_test, y_test)

    ### Random forest results
    RF_test_auc = RandomForest_TrainTest(X_train, y_train, X_test, y_test)


    return LR_test_auc, RF_test_auc


def multiple_run_TrainTest(data_dir, save_csv_dir, random_states, embed_dict):
    LR_test_res = []
    RF_test_res = []

    for random_state in random_states:
        train_dir = data_dir + "/TrainEvalTestData/train_data_" + str(random_state) + ".pkl"
        eval_dir = data_dir + "/TrainEvalTestData/eval_data_" + str(random_state) + ".pkl"
        test_dir = data_dir + "/TrainEvalTestData/test_data_" + str(random_state) + ".pkl"
        # prepare the input data
        with open(train_dir, "rb") as f:
            train_data = pickle.load(f)
        with open(eval_dir, "rb") as f:
            eval_data = pickle.load(f)
        with open(test_dir, "rb") as f:
            test_data = pickle.load(f)

        LR_test_auc, RF_test_auc = get_LR_RF_res_TrainTest(train_data, eval_data, test_data, embed_dict)
        LR_test_res.append(f"{LR_test_auc:.4f}")
        RF_test_res.append(f"{RF_test_auc:.4f}")

    rows = zip(LR_test_res, RF_test_res)

    # Write to a CSV file
    with open(save_csv_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optional: Write a header row
        writer.writerow(['LR_test', 'RF_test'])
        # Write the rows
        writer.writerows(rows)