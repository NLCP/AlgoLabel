from util import fcall, load_dataset, dump_dataset
from preprocessing.text.util import filter_sentence
from gensim.models import Word2Vec

from tqdm import tqdm
import numpy as np


class Embedding(object):

    def __init__(self, args):
        pass

    def pretrain(self):
        pass

    def load(self, args):
        pass


@fcall
def load_word2vec(args, input_type):

    size = args["embeddings"]["framework"]["word2vec"]["size"]
    path = "./data/embeddings/w2v_{}_{}.emb".format(input_type, size)

    args["w2v_{}".format(input_type)] = Word2Vec.load(path)
    emb_matrix  = np.array(args["w2v_{}".format(input_type)].wv.vectors)
    unk_vector  = np.mean(emb_matrix, axis=0)
    pad_vector  = np.zeros(size)

    args["w2v_{}_matrix".format(input_type)]     = np.append(emb_matrix, [unk_vector, pad_vector], axis=0)
    args["w2v_{}_vocab_size".format(input_type)] = args["w2v_{}_matrix".format(input_type)].shape[0]


@fcall
def pretrain_word2vec_text(args, settings):

    sentences = []

    sources = ["train", "dev", "unlabeled"]

    for ds_type in sources:

        dataset = load_dataset("./data/datasets/split/text/{}.json".format(ds_type))
        for sample in tqdm(dataset):

            if args["train"]["only_cf"]:
                if sample["source"] != "codeforces":
                    continue

            for sent in sample["sentences"]:
                target_sentences = sample["sentences"][sent]
                sentences += [filter_sentence(s) for s in target_sentences]

    np.random.shuffle(sentences)
    print("Num. sentences {}".format(len(sentences)))

    model = Word2Vec(sentences,
                     size=settings["size"],
                     window=settings["window"],
                     min_count=settings["min_count"],
                     workers=settings["workers"])

    path = "./data/embeddings/w2v_text_{}.emb".format(settings["size"])
    model.save(path)
    print("Saved text embeddings for size {}".format(settings["size"]))


@fcall
def pretrain_word2vec_code(args, settings, input_field):

    sources = []
    for ds_type in ["train", "dev", "unlabeled"]:
        dataset = load_dataset("./data/datasets/split/code/{}_complete.json".format(ds_type))

        for sample in dataset:
            if input_field in sample and sample[input_field]:
                sources.append(sample[input_field])

    np.random.shuffle(sources)
    print("Num. samples {}!".format(len(sources)))

    model = Word2Vec(sources,
                     size=settings["size"],
                     window=settings["window"],
                     min_count=settings["min_count"],
                     workers=settings["workers"])

    path = "./data/embeddings/w2v_code_{}_{}.emb".format(input_field, settings["size"])
    model.save(path)


@fcall
def fill_in_ast_paths(args, settings):

    data = {}

    with open("./data/code/code2vec/cpp/path_contexts.csv", "r") as f:

        for line in f:
            tokens              = line.split()
            id                  = tokens[0].split("\\")[-1].split(".cpp")[0]
            starts, paths, ends = [], [], []

            for path in tokens[1:]:
                start, path, end = path.split(",")
                starts.append(start)
                paths.append(path)
                ends.append(end)

            data[id] = {
                "starts": starts,
                "paths": paths,
                "ends": ends
            }

            if len(starts) == 0 or len(paths) == 0 or len(ends) == 0:
                print(id)
                # exit(0)

            # print(id, data[id])
            # exit(0)

    for ds_type in ["train", "dev", "test"]:

        dataset = load_dataset("./data/datasets/split/code/{}_complete.json".format(ds_type))

        for sample in dataset:

            if sample["index"] not in data:
                print("Sample not found", sample["index"])
                exit(0)
                return

            for field in data[sample["index"]]:
                sample[field] = data[sample["index"]][field]

        dataset = [sample for sample in dataset if
                   len(sample["starts"]) > 0 and
                   len(sample["paths"]) > 0 and
                   len(sample["ends"]) > 0 and
                   len(sample["safe"]) > 0 and
                   len(sample["tokens"]) > 0]

        dump_dataset("./data/datasets/split/code/{}_all.json".format(ds_type), dataset)


def pretrain_embeddings(args):

    scenario   = args["embeddings"][args["pretrain"]["scenario"]]
    settings   = args["embeddings"]["framework"][scenario["emb_type"]]
    input_type = scenario["input_type"]

    if input_type == "text":
        pretrain_word2vec_text(args, settings)
    elif input_type == "tokens" or input_type == "symbs":
        pretrain_word2vec_code(args, settings, input_type)
    elif input_type == "ast":
        fill_in_ast_paths(args, settings)