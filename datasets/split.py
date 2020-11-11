import logging
from util import load_dataset, dump_dataset, fcall, multi_process, print_defaultdict
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint as pp
import numpy as np
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification


def store_sample(distribution, sample, difficulty):
    distribution[difficulty].append(sample)
    sample["difficulty_class"] = difficulty


@fcall
def filter_irrelevant_tasks(args, dataset):
    main_ds        = []
    aux_ds         = []
    dups           = set()
    fields         = args["preprocess"]["text"]["fields"]
    ignore_phrases = [
        "This problem was deleted from the contest",
        "This is an interactive"
    ]

    def check_valid_submissions(sample):
        if "solutions" in sample and len(sample["solutions"]) > 0:
            aux_ds.append(sample)

    for sample in dataset:

        if "statement" not in sample:
            check_valid_submissions(sample)
            continue

        ignore = False
        for phrase in ignore_phrases:
            if phrase in sample["statement"]:
                break
        if ignore:
            check_valid_submissions(sample)
            continue

        if "tags" not in sample:
            check_valid_submissions(sample)
            continue

        if "interactive" in sample["tags"]:
            check_valid_submissions(sample)
            continue

        if "contest_name" in sample and "April Fools" in sample["contest_name"]:
            continue

        if not "sentences" in sample:
            check_valid_submissions(sample)
            continue

        check = False
        for field in fields:
            if fields[field]:
                if len(sample["sentences"][field]) == 0:
                    check = True
                    break
        if check:
            check_valid_submissions(sample)
            continue

        uid = "{}_{}".format(sample["index"], sample["source"])
        if uid in dups:
            continue
        dups.add(uid)

        main_ds.append(sample)

    return aux_ds, main_ds


@fcall
def compute_tag_distribution(args, dataset):
    label_distro = defaultdict(int)
    logging.info("Num. of samples ".format(len(dataset)))

    for sample in dataset:
        for tag in sample["tags"]:
            if tag in args["split"]["labels"]:
                label_distro[tag] += 1

    print_defaultdict(label_distro)
    return label_distro


def split_stratified(args, dataset):
    Y = np.array([sample["Y"] for sample in dataset])
    dataset = np.array(dataset)

    percentage = args["split"]["percentage"]
    stratifier = IterativeStratification(n_splits=2, order=2,
                                         sample_distribution_per_fold=[percentage, 1.0 - percentage],
                                         random_state=42)
    remaining_idx, test_idx = next(stratifier.split(dataset, Y))

    X_test  = dataset[test_idx]
    dataset = dataset[remaining_idx]
    Y       = Y[remaining_idx]

    percentage = percentage / (1.0 - percentage)
    stratifier = IterativeStratification(n_splits=2, order=2,
                                         sample_distribution_per_fold=[percentage, 1.0 - percentage])

    train_idx, dev_idx = next(stratifier.split(dataset, Y))

    X_train = dataset[train_idx]
    X_dev   = dataset[dev_idx]

    return list(X_train), list(X_dev), list(X_test)


def store_sample_percentage(difficulty_distro,
                            distribution,
                            difficulty_classes,
                            percentage,
                            targets):
    new_dataset = []

    for difficulty in difficulty_classes:

        count_ds    = difficulty_distro[difficulty]
        samples_ds  = distribution[difficulty]
        old_dataset = []

        Y = []
        for sample in samples_ds:
            y = []
            for label in targets:
                if label in sample["tags"]:
                    y.append(1)
                else:
                    y.append(0)
            sample["Y"] = y
            Y.append(y)

        if percentage == 1.0:
            X_test = samples_ds
            X_train = []
        else:
            samples_ds = np.array(samples_ds)
            Y          = np.array(Y)

            stratifier = IterativeStratification(n_splits=2, order=2,
                                                 sample_distribution_per_fold=[percentage, 1.0 - percentage],
                                                 random_state=42)
            train_indexes, test_indexes = next(stratifier.split(samples_ds, Y))

            # print(train_indexes.shape, test_indexes.shape)
            # print(len(train_indexes), len(test_indexes))
            #
            # print(samples_ds.shape)
            # print(Y.shape)

            print(train_indexes[:10])
            print(test_indexes[:10])

            X_train = samples_ds[train_indexes]
            X_test  = samples_ds[test_indexes]

            # exit(0)
            # X_train, _ = samples_ds[train_indexes], Y[train_indexes, :]
            # X_test, _  = samples_ds[test_indexes], Y[test_indexes, :]

        distribution[difficulty] = list(X_train)
        new_dataset += list(X_test)

        # target_count  = {label: int(percentage * count_ds[label]) for label in targets}
        # current_count = {label: 0 for label in targets}
        #
        # for sample in sampels_ds:
        #
        #     ok = False
        #     for label in targets:
        #         if label in sample["tags"]:
        #             if target_count[label] > current_count[label]:
        #                 ok = True
        #                 break
        #
        #     if ok:
        #         Y = []
        #         for label in targets:
        #             if label in sample["tags"]:
        #                 current_count[label] += 1
        #                 Y.append(1)
        #             else:
        #                 Y.append(0)
        #         sample["Y"] = Y
        #
        #         new_dataset.append(sample)
        #     else:
        #         old_dataset.append(sample)
        # distribution[difficulty] = old_dataset

    return new_dataset


def strip_text_sample(sample):
    return {key: sample[key] for key in sample if key != "solutions"}


def extract_solutions(dataset):
    result = []
    for sample in dataset:
        if "solutions" in sample:
            for solution in sample["solutions"]:
                if "Y" in sample:
                    solution["Y"]    = sample["Y"]
                if "tags" in sample:
                    solution["tags"] = sample["tags"]
                result.append(solution)
    return result


def setup_Y_field(dataset, targets):
    for sample in dataset:
        y = []
        for label in targets:
            if label in sample["tags"]:
                y.append(1)
            else:
                y.append(0)
        sample["Y"] = y


@fcall
def split_dataset(args):
    params = args["split"]

    dataset      = load_dataset("./data/datasets/dataset.json")
    distribution = defaultdict(list)
    targets      = args["split"]["labels"]

    np.random.seed(params["random_seed"])
    np.random.shuffle(dataset)

    splits   = ["train", "dev", "test", "unlabeled"]
    datasets = {
        "code": {split: [] for split in splits},
        "text": {split: [] for split in splits}
    }

    aux_ds, main_ds = filter_irrelevant_tasks(args, dataset)
    """
        aux_ds  contains code samples that do not have an associated text sample
        main_ds comprises samples with both text and code input
    """

    def check_valid_solutions(sample):
        if "solutions" in sample and len(sample["solutions"]) > 0:
            datasets["code"]["unlabeled"].append(sample)

    def check_valid_statement(sample):
        if "statement" in sample and len(sample["statement"]) > 0:
            datasets["text"]["unlabeled"].append(sample)

    def check_valid_labels(sample):
        if "tags" not in sample:
            return False
        for tag in sample["tags"]:
            if tag in targets:
                return True
        return False

    for sample in aux_ds:
        if check_valid_labels(sample):
            datasets["code"]["train"].append(sample)
        else:
            datasets["code"]["unlabeled"].append(sample)

    #################################################################

    text_ds = []
    """text_ds comprises samples with textual input and relevant labels"""

    for sample in main_ds:
        if check_valid_labels(sample):
            text_ds.append(sample)
        else:
            # if submission is valid, we can add the sample to the unlabeled set"
            check_valid_solutions(sample)
            check_valid_statement(sample)

    #################################################################
    # Split dataset based on difficulty

    for sample in text_ds:

        if "difficulty" not in sample or sample["difficulty"] is None:
            store_sample(distribution, sample, "Various")
            continue

        diff = int(sample["difficulty"])
        if diff <= 1500:
            store_sample(distribution, sample, "Easy")
        elif 1500 < diff < 2500:
            store_sample(distribution, sample, "Medium")
        else:
            store_sample(distribution, sample, "Hard")

    #################################################################
    setup_Y_field(text_ds, targets)

    def analyze_diff_distribution(ds, label):
        logging.info("Analyze distribution of tasks by difficulty {}.. ".format(label))
        count_diff = defaultdict(int)
        for sample in ds:
            count_diff[sample["difficulty_class"]] += 1
        print_defaultdict(count_diff)

    ds = distribution["Easy"] + distribution["Medium"] + distribution["Hard"]
    np.random.shuffle(ds)

    analyze_diff_distribution(ds, "Original")

    data = {}
    data["train"], data["dev"], data["test"] = split_stratified(args, ds)
    data["train"] += distribution["Various"]

    analyze_diff_distribution(data["train"], "Train")
    analyze_diff_distribution(data["dev"], "Dev")
    analyze_diff_distribution(data["test"], "Test")

    logging.info("\n\n\nFinal Distribution for the text dataset")

    for split in ["test", "dev", "train"]:
        logging.info("\n[split_dataset] Distro for {}".format(split))
        compute_tag_distribution(args, data[split])
        datasets["text"][split] = [strip_text_sample(s) for s in data[split]]

    datasets["text"]["unlabeled"] = [strip_text_sample(s) for s in datasets["text"]["unlabeled"]]

    logging.info("\n\n\n[split_dataset] Final Distribution for the code dataset")

    for split in ["test", "dev", "train"]:
        logging.info("\n[split_dataset] Distro for {}".format(split))
        samples = extract_solutions(data[split])
        if split == "train":
            samples += extract_solutions(datasets["code"]["train"])
        compute_tag_distribution(args, samples)
        datasets["code"][split] = samples

    datasets["code"]["unlabeled"] = extract_solutions(datasets["code"]["unlabeled"])

    logging.info("[split_dataset] Saving results..")
    for ds_type in ["code", "text"]:
        for split in splits:
            dump_dataset("./data/datasets/split/{}/{}.json".format(ds_type, split),
                         datasets[ds_type][split])
