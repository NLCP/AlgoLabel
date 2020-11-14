from pprint import pprint as pp
import re, json, csv
from nltk import word_tokenize
import sys
import time
from datetime import timedelta
import multiprocessing as mp
import logging
import os
import ast
import itertools
import pickle
import torch
import numpy as np
import random
from subprocess import call

tokenize_regex = re.compile("([^a-zA-Z_#$@0-9<>])")
filtered_toks = {'\xa0', '', ' '}
non_ascii_regex = re.compile(r'[^\x00-\x7f]')


def parse_csv(path, encoding="utf8", delimiter="\t"):
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


def fcall(fun):
    def wrapper(*args, **kwargs):
        sys.stdout.write("[{}] ...\n".format(fun.__name__))
        sys.stdout.flush()

        start_time = time.perf_counter()
        res = fun(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        sys.stdout.write("[{}] Done! {}s\n".format(fun.__name__, timedelta(seconds=runtime)))
        sys.stdout.flush()

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


def multi_process(fun, data, args=None, cpus=None):
    if not cpus:
        pool = mp.Pool(int(mp.cpu_count()))
    else:
        pool = mp.Pool(int(cpus))

    if not args:
        results = pool.map(fun, data)
    else:
        results = [pool.apply(fun, args=(sample, *args)) for sample in data]

    pool.close()
    return results


@fcall
def parse_config():
    return load_dataset("project.json")


def load_dataset(path):
    print("Load dataset {}!".format(path))

    if ".csv" in path or ".tsv" in path:
        return parse_csv(path)

    if ".pt" in path:
        data = torch.load(path)
    elif ".bin" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, encoding='utf-8') as f:
            if ".jsonl" in path:
                data = [json.loads(line) for line in f]
            elif ".json" in path:
                data = json.loads(f.read())

    print("Loaded {} records!".format(len(data)))
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

    try:
        dump_data()
    except FileNotFoundError:
        directory_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            dump_data()


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
def parse_config(path="project.json"):
    return load_dataset(path)


def setup_logging(args):
    level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "critical": logging.CRITICAL
    }

    args = args["logging"]
    logging.basicConfig(filename=args["filename"],
                        filemode=args["filemode"],
                        level=level[args["level"]])
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def set_random_seed(args):
    torch.manual_seed(args["random_seed"])
    np.random.seed(int(args["random_seed"]))
    random.seed(args["random_seed"])


def run_system_command(cmd, shell=True, err_msg=None, verbose=True, split=True):
    if verbose:
        sys.stdout.write("System cmd: {}\n".format(cmd))
    if split:
        cmd = cmd.split()
    rc = call(cmd, shell=shell)
    if err_msg and rc:
        sys.stderr.write(err_msg)
        exit(rc)
    return rc


def tf_gpu_housekeeping():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
