import os
os.chdir('/Users/david/Desktop/FinetuneEmbed')
from mod.options import *

def main():
    # Create the parser
    parser = init_parser()
    # Parse arguments
    args = parser.parse_args()

    print(type(args.model_name))

    # Print all arguments
    print("Arguments and their values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

if __name__ == "__main__":
    main()