import logging
from util import load_dataset, dump_dataset, fcall, multi_process, print_defaultdict
from tqdm import tqdm
from collections import defaultdict

from datasets.textparser import preprocess_text_sample
from datasets.codeparser import preprocess_code_samples
from pprint import pprint as pp


@fcall
def preprocess_code(args, dataset):
    new_dataset = []
    for sample in tqdm(dataset):
        new_dataset.append(preprocess_code_samples(sample, args))
    return new_dataset
    # return multi_process(preprocess_code_samples, dataset, args)


@fcall
def preprocess_text(args, dataset):
    new_dataset = []
    for sample in tqdm(dataset):
        new_dataset.append(preprocess_text_sample(sample, args))
    return new_dataset
    # return multi_process(preprocess_text_sample, dataset, args)


@fcall
def preprocess(args, dataset, source):
    logging.info("Processing {} dataset..".format(source))

    # Multiprocessing is buggy on Windows
    # new_dataset = multi_process(preprocess_text_sample, dataset, args)
    # new_dataset = multi_process(preprocess_code_samples, new_dataset, args)

    params      = args["sources"][source]
    new_dataset = []

    for sample in tqdm(dataset):
        if "text" in params["type"]:
            sample = preprocess_text_sample(sample, args)
        if "code" in params["type"]:
            sample = preprocess_code_samples(sample, args)

        sample = consolidate_tags(sample, args)
        new_dataset.append(sample)

    return new_dataset


def consolidate_tags(sample, args):
    """Add relevant tags (e.g. "dfs" tag also implies "graph" label)"""

    if "tags" not in sample:
        return sample

    rules    = args["preprocess"]["tags"]
    new_tags = set()

    for tag in sample["tags"]:
        new_tags.add(tag)
        if tag in rules:
            new_tags.add(rules[tag])

    sample["tags"] = list(new_tags)
    return sample


@fcall
def aggregate_text(args, dataset):
    params = args["preprocess"]["text"]

    for sample in tqdm(dataset):

        if "statement" not in sample:
            continue

        sentences = sample["sentences"]

        res = [sentences[field]
               for field in params["fields"] if params["fields"][field]]

        args["dataset_text"] += res


@fcall
def aggregate_code(args, dataset):
    for sample in tqdm(dataset):

        if "solutions" not in sample or len(sample["solutions"]) == 0:
            continue

        for solution_idx, solution in enumerate(sample["solutions"]):
            args["dataset_code"].append(solution)


def kattis_opensource_license(args, dataset):
    licenses = defaultdict(int)
    new_ds   = []

    for sample in dataset:

        license = sample["license"]
        licenses[license] += 1
        if "Restricted" not in license:
            new_ds.append(sample)

    print_defaultdict(licenses)
    print(len(dataset), len(new_ds))
    return new_ds


def list_of_kattis_problems(uva):
    titles = set()
    for sample in uva:
        if "hint" in sample:
            if "also available" in sample["hint"].lower():
                title = sample["hint"].split()[-1]
                titles.add(title)

    return titles


def filter_kattis_problem_set(kattis, overlapped):
    dataset = []

    for sample in kattis:

        title = sample["url"].split("/")[-1]
        if title in overlapped:
            continue

        license = sample["license"]
        if "Restricted" in license:
            continue

        dataset.append(sample)

    logging.info("Kattis: {}/{} samples left".format(len(dataset), len(kattis)))
    return dataset


@fcall
def preprocess_dataset(args):
    sources = args["sources"]

    args["dataset"]        = []
    args["dataset_text"]   = []
    args["dataset_code"]   = []
    args["formulas_count"] = defaultdict(int)
    args["formulas"]       = {}

    # Load UVA
    uva             = load_dataset("./data/sources/uva.json")
    kattis_overlap  = list_of_kattis_problems(uva)
    args["dataset"] = preprocess(args, uva, "uva")

    # Load Kattis
    kattis = load_dataset("./data/sources/kattis.json")
    kattis = filter_kattis_problem_set(kattis, kattis_overlap)
    args["dataset"] += preprocess(args, kattis, "kattis")

    for source in sources:
        if source in {"kattis", "uva"}:
            continue
        dataset = load_dataset("./data/sources/{}.json".format(source))
        args["dataset"] += preprocess(args, dataset, source)

    logging.info("[preprocess_dataset] Done!")

    logging.info("[preprocess_dataset] Freq. Text Formulas found:")
    # print_defaultdict(data=args["formulas_count"], max_items=50, verbose=True)
    print_defaultdict(data=args["formulas_count"], file="./logs/formulas", verbose=True)
    logging.info("[preprocess_dataset] Number of distinct formulas found: {}\n".format(len(args["formulas_count"])))

    dump_dataset("./data/datasets/dataset.json", args["dataset"])
