import pickle
import os
import csv
import numpy as np
# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')

from mod.mod_text import *
from mod.options import *

### need to specify 
# data_dir = "./data/long_vs_shortTF"
# save_csv_dir = "./res/2024_1127/LongShortTF/long_vs_shortTF_finetune_auc.csv"
# output_path = "./res/2024_1127/LongShortTF/LongShortTF_model_"
# model_name = "intfloat/e5-small-v2"

def main():
    # Create the parser
    parser = init_parser()
    # Parse arguments
    args = parser.parse_args()

    data_dir = args.data_dir
    save_csv_dir = args.csv_dir
    output_path = args.output_path
    model_name = args.model_name
    random_states = args.random_states

    val_auc_ls = []
    test_auc_ls = []

    for random_state in random_states:
        output_dir = output_path  + str(random_state)

        train_dir = data_dir + "/TrainEvalTestData/train_data_" + str(random_state) + ".pkl"
        eva_dir = data_dir + "/TrainEvalTestData/eval_data_" + str(random_state) + ".pkl"
        test_dir = data_dir + "/TrainEvalTestData/test_data_" + str(random_state) + ".pkl"

        # prepare the input data
        with open(train_dir, "rb") as f:
            train_data = pickle.load(f)
        with open(eva_dir, "rb") as f:
            eval_data = pickle.load(f)
        with open(test_dir, "rb") as f:
            test_data = pickle.load(f)
            
        val_auc_scores = []
        
        train_texts, train_labels = train_data['desc'], train_data['labels']
        eval_texts, eval_labels = eval_data['desc'], eval_data['labels']
        test_texts, test_labels = test_data['desc'], test_data['labels']

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        mod_dir, val_auc = one_fold_training(train_texts, train_labels, eval_texts, eval_labels, 
                                        tokenizer, output_dir, 0, model_name, args)
        
        val_auc_scores.append(val_auc)

        # locate the best model
        best_fold_idx = np.argmax(val_auc_scores)
        best_model_dir = output_dir + '/fold_' + str(best_fold_idx + 1)
        best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)

        # create the training data for final training
        train_texts_all = train_texts + eval_texts
        train_labels_all = train_labels + eval_labels

        full_train_dataset = TextDataset(train_texts_all, train_labels_all, tokenizer)

        # Fine-tune the best model on the full training data
        output_dir = output_dir + "/final_model"
        final_trainer = finetune_best_mod(full_train_dataset, best_model, output_dir, args)

        # Evaluate the best model on the test set
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)
        test_auc = pred_test(final_trainer, test_dataset)

        # save the results
        val_auc_ls.append(f"{val_auc:.4f}")
        test_auc_ls.append(f"{test_auc:.4f}")
        rows = zip(val_auc_ls, test_auc_ls)
        # Write to a CSV file
        with open(save_csv_dir, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Optional: Write a header row
            writer.writerow(['val_auc', 'test_auc'])
            # Write the rows
            writer.writerows(rows)


if __name__ == "__main__":
    main()



