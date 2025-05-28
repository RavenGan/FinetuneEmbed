import argparse

def init_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="Initialize arguments for fine-tuning a model.")
    
    # Arguments for data loading and saving
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the input data.")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to save the output csv files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output models.")
    parser.add_argument("--model_name", type=str, default="GIST-small-Embedding-v0", help="Name of the chosen model.")
    parser.add_argument("--random_states", type=int, nargs='+',default=list(range(41, 51)), help="Random indices used to generate data.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification.")
    parser.add_argument("--ROC_save_dir", type=str, required=True, help="Directory to save ROC plots.")

    # Arguments for fine-tuning
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="The evaluation strategy used in training.")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="The checkpoint saving strategy used in training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="The learning rate used in training.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="The number of epoches used in training.")
    parser.add_argument("--max_grad_norm", type=float, default=0.7, help="The maximum gradient norm used in training.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="The warm up ratio used in training.")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="The gradient weight decay used in training.")

    # Arguments for best model final tuning
    parser.add_argument("--final_evaluation_strategy", type=str, default="no", help="The evaluation strategy used in final training.")
    parser.add_argument("--final_save_strategy", type=str, default="no", help="The checkpoint saving strategy used in final training.")
    parser.add_argument("--final_learning_rate", type=float, default=1e-6, help="The learning rate used in final training.")
    parser.add_argument("--final_num_train_epochs", type=int, default=10, help="The number of epoches used in final training.")
    parser.add_argument("--final_max_grad_norm", type=float, default=0.7, help="The maximum gradient norm used in final training.")
    parser.add_argument("--final_weight_decay", type=float, default=0.2, help="The gradient weight decay used in final training.")
    
    
    return parser