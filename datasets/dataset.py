
from util import load_dataset, dump_dataset, print_defaultdict
from collections import defaultdict
import numpy as np
from skmultilearn.model_selection import IterativeStratification
import logging


class Dataset(object):

    def __init__(self, args, data):

        self.args = args
        self.data = data

    def prepare(self):
        raise NotImplementedError()

    def split_data(self, verbose=True):
        '''
        Split dataset in separate training/validation/test datasets.
        :return:
        '''

        params = self.args["split"]
        labels = params["labels"]

        np.random.shuffle(self.data)

        labeled, unlabeled = self.separate_unlabeled_samples(labels)

        if params["difficulty_based"]:
            distribution = self.split_on_difficulty(labeled)
            dataset = distribution["Easy"] + distribution["Medium"] + distribution["Hard"]
            train, dev, test = self.split_stratified(dataset)
            train += distribution["Various"]
        else:
            train, dev, test = self.split_stratified(labeled)

        data_split = {
            "train"    : self.flatten_samples(train),
            "dev"      : self.flatten_samples(dev),
            "test"     : self.flatten_samples(test),
            "unlabeled": self.flatten_samples(unlabeled)
        }

        if verbose:
            for split in data_split:
                if split != "unlabeled":
                    logging.info("Stats for the {} data split:".format(split))
                    self.compute_tag_distribution(data_split[split])

        return data_split

    def flatten_samples(self, dataset):
        """
            Remove irrelevant fields from samples and
            create separate distinct samples if necessary
            (e.g. list<list<solutions>> -> list<solutions>)
        """
        raise NotImplementedError()

    def serialize(self, ds_path=None):
        dump_dataset(ds_path, self.data)

    def deserialize(self, ds_path):
        self.data = load_dataset(ds_path)

    @staticmethod
    def split_check_relevant_labels(sample, labels):

        if "tags" not in sample:
            return False

        for tag in sample["tags"]:
            if tag in labels:
                return True

        return False

    def separate_unlabeled_samples(self, labels):

        labeled, unlabeled = [], []

        for sample in self.data:
            if self.split_check_relevant_labels(sample, labels):
                labeled.append(sample)
            else:
                unlabeled.append(sample)

        for sample in labeled:
            targets = [1 if label in sample["tags"] else 0
                       for label in labels]
            sample["Y"] = targets

        return labeled, unlabeled

    @staticmethod
    def split_on_difficulty(data):

        distribution = defaultdict(list)

        def store_sample(sample, difficulty_class):
            distribution[difficulty_class].append(sample)
            sample["difficulty_class"] = difficulty_class

        for sample in data:

            if "difficulty" not in sample or \
               sample["difficulty"] is None:
                store_sample(sample, "Various")
                continue

            diff = int(sample["difficulty"])
            if diff <= 1500:
                store_sample(sample, "Easy")
            elif 1500 < diff < 2500:
                store_sample(sample, "Medium")
            else:
                store_sample(sample, "Hard")

        return distribution

    def split_stratified(self, dataset):

        Y = np.array([sample["Y"] for sample in dataset])
        dataset = np.array(dataset)

        percentage = self.args["split"]["percentage"]
        stratifier = IterativeStratification(n_splits=2, order=2,
                                             sample_distribution_per_fold=[percentage, 1.0 - percentage])
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

    def compute_tag_distribution(self, dataset):

        label_distro = defaultdict(int)
        logging.info("Num. of samples ".format(len(dataset)))

        for sample in dataset:
            for tag in sample["tags"]:
                if tag in self.args["split"]["labels"]:
                    label_distro[tag] += 1

        print_defaultdict(label_distro)
        return label_distro
