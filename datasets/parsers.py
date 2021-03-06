import logging
from models.embeddings.word2vec import Word2VecEmbedding
from keras.preprocessing import sequence
import numpy as np


class FeatureParser(object):

    def __init__(self, args, scenario_params, encoder):
        self.args        = args
        self.encoder     = encoder
        self.input_type  = scenario_params["input"]
        self.input_field = scenario_params["field"]

    def _fit_input(self, input_field):
        raise NotImplementedError()

    def filter_incompatible_samples(self, samples):
        return [x for x in samples if self.input_field in x]

    def extract_features(self, samples):

        X, X_meta = [], []

        for sample in samples:

            if self.input_field not in sample:
                raise Exception("{} field not computed for sampled {}"
                                .format(self.input_field, sample["index"]))

            result = self._fit_input(sample[self.input_field])

            if isinstance(result, tuple):
                x, x_meta = result
                X_meta.append(x_meta)
            else:
                x = result

            X.append(x)

        X = self._pad_input(X)

        if len(X_meta) > 0:
            return X, X_meta
        else:
            return X

    def _pad_input(self, X):

        max_seq_len = self.encoder["max_seq_len"]
        sample_size = self.encoder["default_input_size"]
        dtype       = self.encoder["dtype"]

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

                new_x[-len(x):] = x
                X[idx] = new_x.tolist()
            else:
                X[idx] = x[-max_seq_len:]

        return X


class Word2VecParser(FeatureParser):

    features_kind = "Word2Vec"

    def __init__(self, args, parser_params, encoder):
        super().__init__(args, parser_params, encoder)

        self.embeddings = Word2VecEmbedding(args,
                                            input_type=self.input_type,
                                            input_field=self.input_field)
        self.embeddings.load_weights()

    def _fit_input(self, input_field):

        x, x_tokens = [], []

        for token in input_field:
            token_id = self.embeddings.get_token_id(token)
            x.append(token_id)

            if token_id == self.embeddings.unk_idx:
                x_tokens.append("[{}]".format(token))
            else:
                x_tokens.append(token)

        return x, x_tokens

    def _pad_input(self, X):

        max_seq_len = self.encoder["max_seq_len_{}".format(self.input_type)]
        pad_idx     = self.embeddings.pad_idx
        X = sequence.pad_sequences(X, maxlen=max_seq_len, value=pad_idx).tolist()
        return X


class Code2VecParser(FeatureParser):

    features_kind = "Code2Vec"

    def __init__(self, args, parser_params, encoder):
        super().__init__(args, parser_params, encoder)
        self.params           = args["features"]["types"]["code2vec"]
        self.ignore_path_ends = self.params["ignore_path_ends"]

    def _fit_input(self, input_field):

        return [
            input_field["start"],
            input_field["path"],
            input_field["end"]
        ]

    def extract_features(self, samples):

        X = {
            "start": [],
            "path": [],
            "end": []
        }

        for sample in samples:

            if self.input_field not in sample:
                raise Exception("{} field not computed for sampled {}"
                                .format(self.input_field, sample["index"]))

            paths = {
                "start": [],
                "path": [],
                "end": []
            }

            for entry in sample[self.input_field]:

                path_index = int(entry["path_index"])
                if path_index == -1:
                    path_index = self.encoder["path_vocab_size"]

                start_index = int(entry["start"])
                if start_index == -1:
                    start_index = self.encoder["token_vocab_size"]

                end_index = int(entry["end"])
                if end_index == -1:
                    end_index = self.encoder["token_vocab_size"]

                paths["start"].append(start_index)
                paths["path"].append(path_index)
                paths["end"].append(end_index)

            for field in paths:
                X[field].append(paths[field])

        for field in X:
            X[field] = self._pad_input(X[field])

        return {
            0: X["start"],
            1: X["path"],
            2: X["end"]
        }


class SafeParser(FeatureParser):

    features_kind = "SAFE"

    def __init__(self, args, parser_params, encoder):
        super().__init__(args, parser_params, encoder)

    def _fit_input(self, input_field):
        return input_field












