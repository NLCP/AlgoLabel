import argparse
import logging

from util import parse_config, setup_logging, set_random_seed
from models.utils import preprocess_dataset, split_dataset, prepare_embeddings
from models.utils import prepare_input, train, test


def main():

    parser = argparse.ArgumentParser()
    arguments = [
        ("preprocess", preprocess_dataset, "Preprocess samples - cleaning/filtering of invalid data."),
        ("split", split_dataset, "Split dataset in separate folds for training/validation/testing."),
        ("pretrain", prepare_embeddings, "Precompute input representations from unlabeled/training data."),
        ("prepare_input", prepare_input, "Convert raw inputs to numpy compatible data types."),
        ("train", train, "Train currently selected model."),
        ("test", test, "Run available model on evaluation data.")
        # ("analyse", analyse_dataset),                 # WIP
        # ("extract_embeddings", extract_embeddings),   # WIP
    ]

    for arg, _, description in arguments:
        parser.add_argument('--{}'.format(arg), action='store_true', help=description)

    params = parser.parse_args()
    args   = parse_config("config.json")

    setup_logging(args)
    set_random_seed(args)

    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            logging.info("Performing {} operation..".format(arg))
            fun(args)


if __name__ == "__main__":
    main()
