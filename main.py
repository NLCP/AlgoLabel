import argparse
import logging

from datasets.preprocess import preprocess_dataset
from datasets.split import split_dataset
from datasets.pretrain import pretrain_embeddings
from datasets.prepare import prepare_input
from datasets.analyse import analyse_dataset
from models.utils import train, test, extract_embeddings
import sys

from util import parse_config, setup_logging, set_random_seed


def main():
    parser = argparse.ArgumentParser()

    arguments = [
        ("preprocess", preprocess_dataset),
        ("split", split_dataset),
        ("analyse", analyse_dataset),
        ("pretrain", pretrain_embeddings),
        ("prepare_input", prepare_input),
        ("train", train),
        ("test", test),
        ("extract_embeddings", extract_embeddings),
    ]

    for arg, _ in arguments:
        parser.add_argument('--{}'.format(arg), action='store_true')

    params = parser.parse_args()
    args = parse_config("./config.json")

    setup_logging(args)
    set_random_seed(args)

    for arg, fun in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            logging.info("Performing {} operation..".format(arg))
            fun(args)


if __name__ == "__main__":
    main()
