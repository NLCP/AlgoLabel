from util import fcall, load_dataset, dump_dataset
from gensim.models import Word2Vec

from tqdm import tqdm
import numpy as np

from preprocessing.text.util import filter_sentence
from datasets.pretrain import load_word2vec
from keras.preprocessing import sequence
from collections import defaultdict


def process_sentence(args, X, token_X, sent, emb_type):
    embs = args["w2v_{}".format(emb_type)]
    unk_idx = len(args["w2v_{}_matrix".format(emb_type)]) - 2

    for token in sent:
        if token in embs:
            X.append(int(embs.wv.vocab.get(token).index))
            token_X.append(token)
        else:
            X.append(unk_idx)
            token_X.append("[{}]".format(token))


def fit_input_for_w2v_text(args, sample, input):
    if "w2v_text" not in args:
        load_word2vec(args, "text")

    field = input["field"]

    if field == "all":
        sentences = sample["sentences"]["statement"] + \
                    sample["sentences"]["input"] + \
                    sample["sentences"]["output"]
    else:
        sentences = sample["sentences"][field]

    sentences = [filter_sentence(x) for x in sentences]

    X, X_tokens = [], []

    for sentence in sentences:
        process_sentence(args, X, X_tokens, sentence, "text")

    return X, X_tokens


def fit_input_for_w2v_code(args, sample, input):
    field = input["field"]
    tokens = sample[field]
    input_type = "code_{}".format(field)

    if "w2v_{}".format(input_type) not in args:
        load_word2vec(args, input_type)

    X, X_tokens = [], []
    process_sentence(args, X, X_tokens, tokens, input_type)

    return X, X_tokens


def pad_input_for_w2v(args, X, input, input_type):
    encoder = input["encoder"]
    encoder = args["encoders"][encoder]
    max_seq_len = encoder["max_seq_len_{}".format(input_type)]
    pad_idx = len(args["w2v_{}_matrix".format(input_type)]) - 1

    X = sequence.pad_sequences(X,
                               maxlen=max_seq_len,
                               value=pad_idx).tolist()
    return X


def pad_input(args, X, input):
    encoder = input["encoder"]
    encoder = args["encoders"][encoder]
    max_seq_len = encoder["max_seq_len"]
    sample_size = encoder["default_input_size"]
    dtype = encoder["dtype"]

    for idx, x in enumerate(X):

        if len(x) == max_seq_len:
            continue

        if len(x) < max_seq_len:
            if sample_size == 1:
                new_x = np.full(shape=(max_seq_len,),
                                fill_value=0,
                                dtype=dtype)
            else:
                new_x = np.full(shape=(max_seq_len, sample_size),
                                fill_value=0,
                                dtype=dtype)
            # print("idx", idx)
            # print("x", x)
            # print("len", len(x))
            # print("shape", max_seq_len, sample_size)
            new_x[-len(x):] = x
            X[idx] = new_x.tolist()
        else:
            X[idx] = x[-max_seq_len:]

    # print(input_type, ": Padded input shape", X.shape, X[0].shape)
    return X


@fcall
def fit_input(args, dataset):
    model_name = args["model"]
    model = args["models"][model_name]
    encoders = model["encoders"]
    input_type = encoders["input_type"]  # text / code
    inputs = encoders["inputs"]

    X, X_toks = defaultdict(list), defaultdict(list)
    Y = []

    unlabeled = 0

    for sample in tqdm(dataset):

        if "Y" not in sample:
            y = []
            bad = True
            for label in args["split"]["labels"]:
                if label in sample["tags"]:
                    y.append(1)
                    bad = False
                else:
                    y.append(0)
            sample["Y"] = y
            if bad:
                unlabeled += 1
                continue

        Y.append(sample["Y"])

        for input in inputs:

            scenario = args["embeddings"][input["scenario"]]
            emb_type = scenario["emb_type"]
            input_type = scenario["input_type"]

            if emb_type == "safe":
                X[input["field"]].append(sample[input["field"]])
            elif emb_type == "ast":
                for field in ["starts", "paths", "ends"]:
                    X[field].append(sample[field])
            else:
                if emb_type == "word2vec":
                    if input_type == "text":
                        x, toks = fit_input_for_w2v_text(args, sample, input)
                        X[input["field"]].append(x)
                        X_toks[input["field"]].append(toks)
                    elif input_type == "tokens" or input_type == "symbs":
                        x, toks = fit_input_for_w2v_code(args, sample, input)
                        X[input["field"]].append(x)
                        X_toks[input["field"]].append(toks)
                    else:
                        print("Unrecognized input type", input_type)
                        exit(1)
                        return

    if unlabeled:
        print("Unlabeled samples!!!", unlabeled)

    res_X, res_X_toks = [], []
    for input in inputs:

        scenario = args["embeddings"][input["scenario"]]
        input_type = scenario["input_type"]

        if input["scenario"] == "code2vec":
            for field in ["starts", "paths", "ends"]:
                x = X[field]
                res = pad_input(args, x, input)
                res_X.append(res)
            continue

        x = X[input["field"]]

        if input["field"] == "safe":
            res = pad_input(args, x, input)
        elif input_type == "text":
            res = pad_input_for_w2v(args, x, input, "text")
        elif input_type == "tokens" or input_type == "symbs":
            res = pad_input_for_w2v(args, x, input, "code_{}".format(input["field"]))
        else:
            print("Unrecognized input field")
            exit(1)
            return

        res_X.append(res)
        if input["field"] != "safe":
            toks = X_toks[input["field"]]
            res_X_toks.append(toks)

    # res_X = np.array(res_X).tolist()
    return (res_X, res_X_toks), Y


@fcall
def prepare_extra_supervision(args, dataset, source_embeddings):
    problem_emb = source_embeddings["problem_emb"]
    label_emb = source_embeddings["label_emb"]

    Y = []
    for sample in tqdm(dataset):

        if sample["index"] in problem_emb:
            Y.append(problem_emb[sample["index"]])
        else:
            y = []
            for tag in sample["tags"]:
                if tag in label_emb:
                    y.append(label_emb[tag])

            if len(y) == 0:
                print("Tag missing", sample["index"], sample["tags"])
                exit(1)

            y = [float(x) for x in (np.average(y, axis=0))]
            Y.append(y)

    return Y


@fcall
def keep_only_cf_sample(dataset):
    new_dataset = []
    for sample in dataset:
        if "source" in sample:
            if sample["source"] == "codeforces":
                new_dataset.append(sample)
        else:
            if "_" not in sample["index"]:
                print("Index", sample["index"])
                exit(0)
            source = sample["index"].split("_")[0]
            if source == "codeforces":
                new_dataset.append(sample)

    return new_dataset


@fcall
def join_cf_dataset(args, ds_code, ds_text):
    problems = {}

    for sample in tqdm(ds_code):

        if not "code" in sample or not sample["code"] or len(sample["code"]) < 10:
            continue

        if not "safe" in sample or not sample["safe"] or len(sample["safe"]) == 0:
            continue

        if not "starts" in sample or not sample["starts"] or len(sample["starts"]) == 0:
            continue

        sample["submission"] = sample["index"]
        sample["index"] = sample["index"].split("_")[1]
        problems[sample["index"]] = sample

    dataset = []
    without_valid_code = 0
    without_valid_statement = 0

    for sample in tqdm(ds_text):

        if not "statement" in sample or not sample["statement"]:
            without_valid_statement += 1
            continue

        if not sample["index"] in problems:
            without_valid_code += 1
            continue

        problem = problems[sample["index"]]

        for key in sample:
            if key not in problem:
                problem[key] = sample[key]

        dataset.append(problem)

    print("Without valid statement", without_valid_statement)
    print("Without valid code", without_valid_code)
    print("New dataset len", len(dataset))
    return dataset


@fcall
def prepare_input(args):
    model_name = args["model"]
    model = args["models"][model_name]
    encoders = model["encoders"]
    input_type = encoders["input_type"]  # text / code

    for split in ["train", "dev", "test"]:

        if input_type == "both":
            dataset_text = load_dataset("./data/datasets/split/text/{}.json".format(split))
            dataset_code = load_dataset("./data/datasets/split/code/{}.json".format(split))

            dataset_code = keep_only_cf_sample(dataset_code)
            dataset_text = keep_only_cf_sample(dataset_text)

            dataset = join_cf_dataset(args, dataset_code, dataset_text)
        else:
            path = split
            dataset = load_dataset("./data/datasets/split/{}/{}.json".format(input_type, path))
            if args["train"]["only_cf"]:
                dataset = keep_only_cf_sample(dataset)

        X, Y = fit_input(args, dataset)
        X, X_toks = X

        dump_dataset("./data/models/{}/data/X_{}.json".format(model_name, split), X)
        dump_dataset("./data/models/{}/data/X_toks_{}.json".format(model_name, split), X_toks)
        dump_dataset("./data/models/{}/data/Y_{}.json".format(model_name, split), Y)

        if input_type == "text" and args["train"]["extra_supervision"]:
            source_embeddings = load_dataset("./data/embeddings/source_emb_{}.json".format(split))
            Y_extra = prepare_extra_supervision(args, dataset, source_embeddings)
            dump_dataset("./data/models/{}/data/Y_{}_source_emb.json".format(model_name, split),
                         Y_extra)
