
from util import load_dataset, dump_dataset, fcall, multi_process
from util import ensure_path

from models.classifiers.algolabel import AlgoLabel

from datasets.source_dataset import SourceDataset
from datasets.text_dataset   import TextDataset
from datasets.source_dataset import prepare as source_prepare
from datasets.source_dataset import save_code2vec_index, match_ast_data

from models.embeddings.word2vec import Word2VecEmbedding
from pathlib import Path
import logging

from safe.safe import prepare_dataset as compute_safe_embeddings
from safe.safe import SAFE
from datasets.pipeline import SourcePipeline
import numpy as np
import os


def identify_model(args):

    type_map = {
        "AlgoSourceLabel": AlgoLabel
    }

    try:
        return type_map[args["model"]](args)
    except KeyError:
        raise NotImplementedError("Unrecognized model type {}!".format(args["model"]))


def identify_data_handler(args):

    handler_map = {
        "SourceDataset": SourceDataset,
        "TextDataset": TextDataset
    }

    try:
        return handler_map[args["prepare"]["handler"]]
    except KeyError:
        raise NotImplementedError("Unrecognized data handler type {}!".format(args["model"]))


def extract_root_path(path):
    return path.split(".json")[0]


def prepare_dataset_parallel(args, dataset):

    if args["prepare"]["handler"] == "SourceDataset":

        handler = SourceDataset(args, dataset)
        dataset = handler.filter_irrelevant_tasks()
        logging.debug("Preparing sources for {} problems..".format(len(dataset)))

        dataset = multi_process(source_prepare,
                                dataset,
                                args,
                                cpus=4,
                                batch_size=args["prepare"]["batch_size"])
    else:
        raise NotImplementedError("Handler not available")

    dataset = [sample for sample in dataset if sample]
    return dataset


@fcall
def preprocess_dataset(args):

    dataset = load_dataset(args["raw_dataset"])
    handler = identify_data_handler(args)

    if args["prepare"]["parallel"]:
        dataset = prepare_dataset_parallel(args, dataset)
    else:
        # Custom dataset preprocessing module
        dataset = handler(args, dataset).prepare()

    root_path = extract_root_path(args["raw_dataset"])
    new_path  = "{}_prepared.json".format(root_path)
    handler(args, dataset).serialize(new_path)


@fcall
def split_dataset(args):

    root_path = extract_root_path(args["raw_dataset"])
    new_path  = "{}_prepared.json".format(root_path)
    dataset   = load_dataset(new_path)

    # Custom dataset preprocessing module
    handler = identify_data_handler(args)
    dataset = handler(args, dataset)

    # Split train/dev/test
    data_split = dataset.split_data(verbose=True)

    for key in data_split:
        split_path = "{}_{}.json".format(root_path, key)
        dump_dataset(split_path, data_split[key])


def _prepare_word2vec(args, params):

    """
        Pretrain Word2Vec embeddings for the specified input field(s)
    """
    embedding = Word2VecEmbedding(args,
                                  input_type=params["input"],
                                  input_field=params["field"])
    root_path = extract_root_path(args["raw_dataset"])

    train_path = "{}_train.json".format(root_path)
    train = load_dataset(train_path)

    unlab_path = "{}_unlabeled.json".format(root_path)
    unlabeled = load_dataset(unlab_path)

    X = train + unlabeled
    embedding.pretrain(X)


def _prepare_code2vec(args, params):

    root_path = extract_root_path(args["raw_dataset"])
    path_contexts = {}

    datasets = {}

    def compute_fold_path(label):
        return "{}_{}.json".format(root_path, label)

    for fold in ["dev", "test", "train"]:
        fold_path = compute_fold_path(fold)
        datasets[fold] = load_dataset(fold_path)

        destination = Path.cwd() / "data" / "code" / fold

        dataset = SourceDataset(args, datasets[fold])
        result = dataset.preprocess_ast(path=destination,
                                        force_rewrite=False)
        path_contexts[fold] = result

    train_data = path_contexts["train"]
    for fold in ["dev", "test"]:
        fold_data = match_ast_data(path_contexts[fold], train_data)
        dataset   = save_code2vec_index(datasets[fold], fold_data, fold)
        dump_dataset(compute_fold_path(fold), dataset)

    dataset = save_code2vec_index(datasets["train"], train_data, "train")
    dump_dataset(compute_fold_path("train"), dataset)


def _prepare_safe(args, params):

    params    = args["features"]["types"]["safe"]
    safe      = SAFE(params["model"], params["instr_conv"], params["max_instr"])
    root_path = extract_root_path(args["raw_dataset"])

    for ds_type in ["test", "dev", "train"]:
        path = "{}_{}.json".format(root_path, ds_type)
        data = load_dataset(path)
        data = compute_safe_embeddings(args, safe, data)
        dump_dataset(path, data)


@fcall
def prepare_embeddings(args):

    scenario = args["pretrain"]
    params   = args["features"]["scenarios"][scenario]

    if params["type"] == "word2vec":
        _prepare_word2vec(args, params)
    elif params["type"] == "code2vec":
        _prepare_code2vec(args, params)
    elif params["type"] == "safe":
        _prepare_safe(args, params)
    else:
        raise NotImplementedError()


def get_model_path(args):
    res_path = Path.cwd() / "data" / "models" / args["model"] / "data"
    ensure_path(res_path)
    return res_path


@fcall
def prepare_input(args):

    pipeline  = SourcePipeline(args).init_from_model_config()
    res_path  = get_model_path(args)

    root_path = extract_root_path(args["raw_dataset"])
    for ds_type in ["test", "dev", "train"]:
        path  = "{}_{}.json".format(root_path, ds_type)
        if not os.path.exists(path):
            logging.critical("Expected {} file in path {}.".format(ds_type, path))
            exit(1)

        data  = load_dataset(path)
        X, Y  = pipeline.run(data)

        if isinstance(X, tuple):
            X, X_meta = X
            meta_path = res_path / "X_{}_meta.json".format(ds_type)
            dump_dataset(meta_path, X_meta)

        if len(X[0]) != len(Y):
            logging.critical("X num samples - {} \n".format(len(X[0])))
            logging.critical("Y num samples - {} \n".format(len(Y)))
            logging.critical("Total num samples - {}\n".format(len(data)))
            raise Exception("Mismatch between number of input and output samples")

        input_path  = res_path / "X_{}.json".format(ds_type)
        dump_dataset(input_path, X)

        output_path = res_path / "Y_{}.json".format(ds_type)
        dump_dataset(output_path, Y)


@fcall
def setup_model(args):
    model = AlgoLabel(args)
    model.build_model()
    model.load_weights()
    return model


@fcall
def load_model_input(args, fold):

    logging.info("Loading preprocessed {} dataset".format(fold))
    path = get_model_path(args)
    X = load_dataset(path / "X_{}.json".format(fold))
    Y = load_dataset(path / "Y_{}.json".format(fold))
    return X, np.array(Y)


@fcall
def train(args):
    model         = setup_model(args)
    train_dataset = load_model_input(args, "train")
    dev_dataset   = load_model_input(args, "dev")
    model.fit(train_dataset, dev_dataset)

    test_dataset  = load_model_input(args, "test")
    model.score(test_dataset)


@fcall
def test(args):
    model = setup_model(args)
    test_dataset  = load_model_input(args, "test")
    model.score(test_dataset)
