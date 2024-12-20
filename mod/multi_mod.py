import torch
import numpy as np
from typing import Union
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pickle
import csv

def get_NoInstruct_small_Embedding(model, tokenizer, text, mode):
    model.eval()

    assert mode in ("query", "sentence"), f"mode={mode} was passed but only `query` and `sentence` are the supported modes."

    if isinstance(text, str):
        text = [text]

    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

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

def load_data_smallmod(train_dict, eval_dict, test_dict, model_name):
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
    else:
        model = SentenceTransformer(model_name)
        X_train = model.encode(train_text)
        X_test = model.encode(test_text)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return X_train, y_train, X_test, y_test


def LogisticReg_TrainTest(X_train, y_train, X_test, y_test):
    # Initialize the logistic regression model
    log_reg = LogisticRegression()

    log_reg.fit(X_train, y_train)
    y_test_pred_proba = log_reg.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    return test_auc

def LogisticReg_TrainTest_CV(X_train, y_train, X_test, y_test):
    # Define the logistic regression model
    model = LogisticRegression(solver='liblinear')  # use liblinear for small datasets and L1 penalty
    
    # Define a range of hyperparameters for tuning
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    
    # Initialize GridSearchCV with cross-validation (CV) on the training data
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # report the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # load the best model and evaluate on the test data
    best_model = grid_search.best_estimator_
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    return test_auc

def RandomForest_TrainTest(X_train, y_train, X_test, y_test):
    # Initialize the random forest model
    random_forest = RandomForestClassifier()

    random_forest.fit(X_train, y_train)
    y_test_pred_proba = random_forest.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    return test_auc

def RandomForest_TrainTest_CV(X_train, y_train, X_test, y_test):
    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier()
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [25, 50, 100, 200, 400],        # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],            # Maximum depth of each tree
    }
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, 
                               scoring='accuracy', verbose=1, 
                               n_jobs=-1 # Use all available cores
                               )
    # Fit the model with the training data
    grid_search.fit(X_train, y_train)
    # report the best parameters
    print("Best hyperparameters found: ", grid_search.best_params_)

    # Get the best model from the grid search
    best_rf_model = grid_search.best_estimator_
    y_test_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    return test_auc

def get_LR_RF_res_TrainTest(train_dict, eval_dict, test_dict, model_name, do_cv):
    # Load the data
    X_train, y_train, X_test, y_test = load_data_smallmod(train_dict, eval_dict, test_dict, model_name)
    
    if do_cv:
        LR_test_auc = LogisticReg_TrainTest_CV(X_train, y_train, X_test, y_test)
        RF_test_auc = RandomForest_TrainTest_CV(X_train, y_train, X_test, y_test)
    else:
        LR_test_auc = LogisticReg_TrainTest(X_train, y_train, X_test, y_test)
        RF_test_auc = RandomForest_TrainTest(X_train, y_train, X_test, y_test)


    return LR_test_auc, RF_test_auc

def smallmod_multiple_run_TrainTest(data_dir, save_csv_dir, random_states, model_name, do_cv):
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

        LR_test_auc, RF_test_auc = get_LR_RF_res_TrainTest(train_data, eval_data, test_data, model_name, do_cv)
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