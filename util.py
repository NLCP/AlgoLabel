from pprint import pprint as pp
import re
import json
import csv
import sys
import time
from datetime import timedelta
import multiprocessing as mp
import logging
import os
import ast
import itertools
import pickle
import random
from subprocess import call
import shutil
from nltk import word_tokenize
import torch
import numpy as np
from tqdm import tqdm

tokenize_regex = re.compile("([^a-zA-Z_#$@0-9<>])")
filtered_toks  = {'\xa0', '', ' '}
non_ascii_regex = re.compile(r'[^\x00-\x7f]')


def serialize_object(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def generic_serialize(obj, path: str):

    try:
        path = str(path)
        serialize_object(obj, path)
    except FileNotFoundError:
        directory_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            try:
                serialize_object(obj, path)
            except IOError:
                logging.critical("Failure to save object to path {}".format(path))
                exit(1)


def generic_deserialize(path: str):

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except IOError:
        logging.critical("Failure to load model from path " + path)
        exit(1)


def parse_csv(path: str, encoding="utf8", delimiter="\t"):

    dataset = []
    with open(path, encoding=encoding) as f:

        reader = csv.reader(f, delimiter=delimiter)
        keys   = reader.__next__()

        for row in reader:
            sample = {}
            for idx, value in enumerate(row):
                sample[keys[idx]] = value
            dataset.append(sample)

    return dataset


def parse_line_csv(path: str, encoding="utf8", delimiter=","):

    dataset = []

    with open(path, encoding=encoding) as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            dataset.append((row[0], row[1:]))

    return dataset


def fcall(fun):
    """
    Convenience decorator used to measure the time spent while executing
    the decorated function.
    :param fun:
    :return:
    """
    def wrapper(*args, **kwargs):

        logging.info("[{}] ...".format(fun.__name__))

        start_time = time.perf_counter()
        res = fun(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        logging.info("[{}] Done! {}s\n".format(fun.__name__, timedelta(seconds=runtime)))
        return res

    return wrapper


@fcall
def multi_process_dataset(fun, data, fargs=None, huge_data=None):

    if huge_data:
        batch_size  = huge_data
        num_samples = len(data)
        idx         = 0
        batch_idx   = 0

        while idx < num_samples:
            batch = data[idx:idx + batch_size]
            temp  = multi_process(fun, batch, fargs)
            dump_dataset("./data/tmp_batch_{}".format(batch_idx), temp)

            batch_idx += 1
            idx += batch_size

        del data

        dataset = []
        for jdx in range(batch_idx):
            dataset += load_dataset("./data/tmp_batch_{}".format(jdx))
        return dataset

    else:
        return multi_process(fun, data, fargs)


def _multi_process_aux(fun, data, args=None, cpus=None):

    if not cpus:
        pool = mp.Pool(int(mp.cpu_count()))
    else:
        pool = mp.Pool(int(cpus))

    if not args:
        results = pool.map(fun, data)
    else:
        results = [pool.apply(fun, args=(sample, args)) for sample in data]

    pool.close()
    return results


def multi_process(fun, data, args=None, cpus=None, batch_size=None):

    if not batch_size:
        return _multi_process_aux(fun, data, args, cpus)

    start     = 0
    num_items = len(data)
    result    = []

    with tqdm(total=num_items) as progress_bar:

        while start < num_items:
            num_batch_items = min(len(data) - start, batch_size)
            items  = _multi_process_aux(fun, data[start:start + num_batch_items], args, cpus)
            result += items
            start  += num_batch_items
            progress_bar.update(num_batch_items)

    return result


def load_dataset(path: str):
    """
    Load dataset from path.
    Supported formats: ".csv", ".tsv", ".pt", ".bin", ".json" and ".jsonl"
    :param path:
    :return:
    """

    logging.info("Load dataset {}!".format(path))
    path = str(path)

    if ".csv" in path or ".tsv" in path:
        return parse_csv(path)

    if ".pt" in path:
        data = torch.load(path)
    elif ".bin" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif "json" in path:
        with open(path, encoding='utf-8') as f:
            if ".jsonl" in path:
                data = [json.loads(line) for line in f]
            elif ".json" in path:
                data = json.loads(f.read())
    else:
        raise NotImplementedError("Don't know how to load a dataset of this type!")

    logging.info("Loaded {} records!".format(len(data)))
    return data


def cleanup_torch_model(model):
    del model
    torch.cuda.empty_cache()


def dump_dataset(path, data, verbose=True):

    if verbose:
        print("Dump dataset {}: {}!".format(path, len(data)))

    def dump_data():

        if ".pt" in path:
            with open(path, "wb") as f:
                torch.save(data, f)
        elif ".bin" in path:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(path, "w") as f:
                f.write(json.dumps(data, indent=4))

    path = str(path)
    try:
        dump_data()
    except FileNotFoundError:
        directory_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            dump_data()


def remove_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def ensure_path(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def check_float(token):

    try:
        float(token)
        return True
    except ValueError:
        return False


def print_defaultdict(data, max_items=None, verbose=True, reverse=True, file=None):
    data = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)

    if verbose:
        to_print = data[:max_items] if max_items else data
        if not file:
            pp(to_print)
        else:
            with open(file, "w", encoding="utf-8") as f:
                pp(to_print, stream=f)

    return data


def normalize_label(label):
    if not label:
        return label

    diacritics = {("ă", "a"), ("ţ", "t"), ("â", "a"), ("î", "i"), ("ş", "s")}

    label = label.strip().lower()
    for k, v in diacritics:
        label = label.replace(k, v)

    return label


stopwords = {"the", "of", "to", "and", "is", "that", "it", "with"}


def replace_chars(text, char_map):
    for ch in char_map:
        text = text.replace(ch, char_map[ch])
    return text


def filter_sentence(sentence):
    return [x.lower() for x in word_tokenize(sentence)
            if len(x) >= 1 and x.lower() not in stopwords]


def tokenize(sentence):
    return [x.lower() for x in word_tokenize(sentence)]


def compare_ast(node1, node2):
    if not isinstance(node1, str):
        if type(node1) is not type(node2):
            return False
    if isinstance(node1, ast.AST):
        for k, v in list(vars(node1).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, zip(node1, node2)))
    else:
        return node1 == node2


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


@fcall
def parse_config(path: str = "project.json"):
    """
    Parse default configuration file
    :param path:
    :return:
    """
    return load_dataset(path)


def setup_logging(args):
    """
    Redirect logging messages to a dedicated log file.
    :param args:
    :return:
    """

    level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "critical": logging.CRITICAL
    }

    msg_format = '%(asctime)s:%(levelname)s: %(message)s'
    formatter  = logging.Formatter(msg_format, datefmt='%H:%M:%S')
    args       = args["logging"]

    file_handler = logging.FileHandler(args["filename"], mode=args["filemode"])
    file_handler.setLevel(level=level[args["level"]])
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger()
    logger.setLevel(level[args["level"]])


def set_random_seed(args):
    """
    Ensure results are predictable
    :param args:
    :return:
    """
    if "torch" in sys.modules:
        torch.manual_seed(args["random_seed"])
    np.random.seed(int(args["random_seed"]))
    random.seed(args["random_seed"])


def run_system_command(cmd: str,
                       shell: bool = False,
                       err_msg: str = None,
                       verbose: bool = True,
                       split: bool = True,
                       stdout=None,
                       stderr=None) -> int:
    """
    :param cmd: A string with the terminal command invoking an external program
    :param shell: Whether the command should be executed through the shell
    :param err_msg: Error message to print if execution fails
    :param verbose: Whether to print the command to the standard output stream
    :param split: Whether to split the tokens in the command string
    :param stdout: file pointer to redirect stdout to
    :param stderr: file pointer to redirect stderr to
    :return: Return code
    """
    if verbose:
        sys.stdout.write("System cmd: {}\n".format(cmd))
    if split:
        cmd = cmd.split()
    rc = call(cmd, shell=shell, stdout=stdout, stderr=stderr)
    if err_msg and rc:
        sys.stderr.write(err_msg)
        exit(rc)
    return rc


def tf_gpu_housekeeping():
    """
    Ensure tensorflow doesn't hog the available GPU memory.
    :return:
    """
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.critical(str(e))
            exit(1)
