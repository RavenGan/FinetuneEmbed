import torch
import numpy as np
from typing import Union
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
import pandas as pd
import pickle
import csv
import os

def get_NoInstruct_small_Embedding(model, tokenizer, text, mode):
    model.eval()

    assert mode in ("query", "sentence"), f"mode={mode} was passed but only `query` and `sentence` are the supported modes."

    if isinstance(text, str):
        text = [text]

    inp = tokenizer(text, return_tensors="pt", padding=True, 
                    truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**inp)

    # The model is optimized to use the mean pooling for queries,
    # while the sentence / document embedding uses the [CLS] representation.

    if mode == "query":
        vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
        vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
    else:
        vectors = output.last_hidden_state[:, 0, :]

    return vectors

def get_BioBERT_Embedding(model, tokenizer, text):
    model.eval()

    if isinstance(text, str):
        text = [text]

    inp = tokenizer(text, return_tensors="pt", padding=True, 
                    truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inp)

    # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
    # Apply mean pooling (average over the token embeddings)
    embedding = outputs.last_hidden_state.mean(dim=1)  # shape: (batch_size, hidden_size)

    return embedding



def load_data_smallmod(train_dict, eval_dict, test_dict, model_name, do_pca=False, n_PCs=384):
    train_text = train_dict['desc']
    train_labels = train_dict['labels']

    eval_text = eval_dict['desc']
    eval_labels = eval_dict['labels']

    test_text = test_dict['desc']
    test_labels = test_dict['labels']

    # Combine the training and validation data
    train_text = train_text + eval_text
    train_labels = train_labels + eval_labels

    if model_name == "avsolatorio/NoInstruct-small-Embedding-v0":
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        X_train = get_NoInstruct_small_Embedding(model, tokenizer, train_text, mode="sentence")
        X_train = np.array(X_train)
        X_test = get_NoInstruct_small_Embedding(model, tokenizer, test_text, mode="sentence")
        X_test = np.array(X_test)

    elif model_name == "dmis-lab/biobert-base-cased-v1.1":
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        X_train = get_BioBERT_Embedding(model, tokenizer, train_text)
        X_train = np.array(X_train)
        X_test = get_BioBERT_Embedding(model, tokenizer, test_text)
        X_test = np.array(X_test)
        
    else:
        model = SentenceTransformer(model_name)
        X_train = model.encode(train_text)
        X_test = model.encode(test_text)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # If PCA is needed, apply PCA here
    if do_pca:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=n_PCs) # number of components to keep
        X_train_pca = pca.fit_transform(X_train_scaled)

        X_test_pca = pca.transform(X_test_scaled)

        return X_train_pca, y_train, X_test_pca, y_test
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
            y_test_bin = label_binarize(y_test, classes=classes)
            test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
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
            y_test_bin = label_binarize(y_test, classes=classes)
            test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
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
            y_test_bin = label_binarize(y_test, classes=classes)
            test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
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
            y_test_bin = label_binarize(y_test, classes=classes)
            test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
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

def get_LR_RF_res_TrainTest(train_dict, eval_dict, test_dict, model_name, 
                            do_cv, ROC_save_path, do_pca, n_PCs):
    # Load the data
    X_train, y_train, X_test, y_test = load_data_smallmod(train_dict, eval_dict, test_dict, model_name,
                                                          do_pca, n_PCs)
    
    if do_cv:
        LR_test_auc = LogisticReg_TrainTest_CV(X_train, y_train, X_test, y_test, ROC_save_path)
        RF_test_auc = RandomForest_TrainTest_CV(X_train, y_train, X_test, y_test, ROC_save_path)
    else:
        LR_test_auc = LogisticReg_TrainTest(X_train, y_train, X_test, y_test, ROC_save_path)
        RF_test_auc = RandomForest_TrainTest(X_train, y_train, X_test, y_test, ROC_save_path)


    return LR_test_auc, RF_test_auc

def smallmod_multiple_run_TrainTest(data_dir, save_csv_dir, random_states, model_name, 
                                    do_cv, ROC_save_path, do_pca=False, n_PCs=384):
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

        LR_test_auc, RF_test_auc = get_LR_RF_res_TrainTest(train_data, eval_data, test_data, model_name, 
                                                           do_cv, save_path, do_pca, n_PCs)
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