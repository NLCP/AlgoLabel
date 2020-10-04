
from util import fcall, load_dataset, dump_dataset
# from keras.engine.input_layer import Input
# from keras.layers.embeddings import Embedding
from datasets.prepare import load_word2vec
# from keras.initializers import Constant
# import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Masking, Concatenate, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Conv1D, ReLU, GRU
from keras.layers import MaxPool1D, Permute, Flatten
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.engine.input_layer import Input
from keras.regularizers import l2
import tensorflow as tf
from keras import backend as K


from models.attention import AttentionWithContext, CustomAttention #, Code2VecAttentionLayer
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, roc_auc_score, hamming_loss

from keras.backend import squeeze

from logs.utils import save_result_log
import numpy as np
from pprint import pprint as pp
import os

from collections import defaultdict
from krippendorff import alpha

import logging
import tensorflow_addons as tfa


class AlgoLabel(object):

    def __init__(self, args):

        self.args       = args
        self.model_name = args["model"]
        self.params     = args["models"][self.model_name]

        print("Model params")
        pp(self.params)

        self.input_layers = []

        self.inputs     = []
        self.att_scores = []

        self.embbedings = {}
        self.encoders   = {}
        self.attention  = {}

        self.model      = None
        self.callbacks  = []
        self.load_epoch = self.args["load_weights_from_epoch"]

    def setup_encoder(self, encoder_name):

        if encoder_name in self.encoders:
            return self.encoders[encoder_name]

        encoder = self.args["encoders"][encoder_name]

        if encoder_name not in {"lstm", "small_lstm", "gru"}:
            raise NotImplementedError("Unrecognized encoder option {}!".format(encoder_name))

        if encoder["regularizer"]:
            regularizer = l2(encoder["regularizer"])
        else:
            regularizer = None

        return_sequences = encoder["attention"]

        if encoder_name in {"lstm", "small_lstm"}:
            rnn = Bidirectional(
                layer=LSTM(units=encoder["hidden_size"],
                           dropout=encoder["dropout"],
                           kernel_regularizer=regularizer,
                           recurrent_regularizer=regularizer,
                           bias_regularizer=regularizer,
                           return_sequences=return_sequences
                ))
        else:
            rnn = Bidirectional(
                layer=GRU(units=encoder["hidden_size"],
                          dropout=encoder["dropout"],
                          kernel_regularizer=regularizer,
                          recurrent_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          return_sequences=return_sequences
                          )
            )

        self.encoders[encoder_name] = rnn
        return rnn

    def setup_attention(self, encoder_name):

        if encoder_name not in self.attention:
            self.attention[encoder_name] = CustomAttention(name="attention_{}".format(encoder_name))
        return self.attention[encoder_name]

    def setup_ast_input_layer(self, input):

        encoder_name = input["encoder"]
        encoder      = self.args["encoders"][input["encoder"]]
        max_seq_len  = encoder["max_seq_len"]

        path_source_token_input = Input(shape=(max_seq_len,),
                                        name="ast_beginnings",)

        path_input              = Input(shape=(max_seq_len,),
                                        name="ast_paths",)

        path_target_token_input = Input(shape=(max_seq_len,),
                                        name="ast_endings",)

        self.inputs.append(path_source_token_input)
        self.inputs.append(path_input)
        self.inputs.append(path_target_token_input)

        token_embedding_shared_layers = Embedding(
            input_dim=encoder["token_vocab_size"],
            output_dim=encoder["token_emb_size"],
            name="token_embedding",
            mask_zero=True,
            input_length=max_seq_len
        )

        # embedding = Embedding(input_dim=vocab,
        #                       output_dim=emb_params["size"],
        #                       embeddings_initializer=Constant(emb_matrix),
        #                       input_length=max_seq_len)

        # print(path_source_token_input, (type(path_source_token_input)))
        # print(token_embedding_shared_layers, (type(token_embedding_shared_layers)))

        path_source_token_embedded = token_embedding_shared_layers(path_source_token_input)
        path_target_token_embedded = token_embedding_shared_layers(path_target_token_input)

        path_embedding = Embedding(
            input_dim=encoder["path_vocab_size"],
            output_dim=encoder["path_emb_size"],
            name="path_embedding",
            mask_zero=True
        )
        paths_embedded = path_embedding(path_input)

        context_embedded = Concatenate(name="ast_context")([path_source_token_embedded,
                                                            paths_embedded,
                                                            path_target_token_embedded])

        context_embedded = Dropout(encoder["dropout"])(context_embedded)

        context_after_dense = TimeDistributed(
                                Dense(2 * encoder["token_emb_size"] + encoder["path_emb_size"],
                                      use_bias=False,
                                      activation='tanh'))(context_embedded)

        attention = self.setup_attention(encoder_name)
        layer, att = attention(context_after_dense)
        self.att_scores.append(att)

        # print(layer, type(layer))
        # print(self.input_layers)

        return layer

    def setup_safe_input_layer(self, input):

        encoder_name = input["encoder"]
        encoder = self.args["encoders"][input["encoder"]]
        max_seq_len = encoder["max_seq_len"]

        name = "{}_encoder".format(input["field"])
        layer = Input(shape=(max_seq_len, encoder["default_input_size"]),
                      name=name)
        self.inputs.append(layer)

        masking = Masking(mask_value=0,
                          input_shape=(max_seq_len,
                                       encoder["default_input_size"]))
        layer = masking(layer)

        rnn = self.setup_encoder(encoder_name)
        layer = rnn(layer)

        if encoder["attention"]:
            attention = self.setup_attention(encoder_name)
            layer, att = attention(layer)
            self.att_scores.append(att)

        return layer

    def setup_simple_input_layer(self, input):

        encoder_name = input["encoder"]
        encoder = self.args["encoders"][input["encoder"]]
        emb_type = self.args["embeddings"]["w2v_cnn_text"]["emb_type"]
        emb_input = self.args["embeddings"]["w2v_cnn_text"]["input_type"]
        emb_params = self.args["embeddings"]["framework"][emb_type]

        name = "{}_encoder".format(input["field"])
        emb_input_type, matrix_name = self.compute_matrix_name(input, emb_input)
        max_seq_len = encoder["max_seq_len_{}".format(emb_input_type)]

        layer = Input(shape=(max_seq_len,), name=name)
        self.inputs.append(layer)

        emb = self.setup_embedding(input, emb_input_type, matrix_name)
        layer = emb(layer)

        return layer

    @staticmethod
    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return focal_loss_fixed

    def build_emnlp_model(self):

        inputs  = self.params["encoders"]["inputs"]
        encoder = self.args["encoders"]["cnn"]

        # setup input layers
        for input in inputs:
            layer = self.setup_simple_input_layer(input)
            self.input_layers.append(layer)

        layer = Concatenate()(self.input_layers)

        convs = [
            ReLU()(Conv1D(filters=encoder["kernel_num"],
                          kernel_size=ks,
                          )(layer))
            for ks in encoder["kernel_sizes"]
        ]

        convs = [
            MaxPool1D()(cv) for cv in convs
        ]

        convs = [
            Permute((2, 1))(cv) for cv in convs
        ]

        layer      = Concatenate(axis=2)(convs)
        layer      = Flatten()(layer)

        ###########################################
        if self.params["encoders"]["joint_encoders"]:
            print("here!?")
            algonet_inputs = self.args["models"]["AlgoNetText"]["encoders"]["inputs"]
            rnn_inputs = []

            emb_input_type, matrix_name = self.compute_matrix_name(inputs[0], "text")
            max_seq_len = encoder["max_seq_len_{}".format(emb_input_type)]
            pad_idx = len(self.args["{}_matrix".format(matrix_name)]) - 1

            for input in self.input_layers:
                masking = Masking(mask_value=pad_idx, input_shape=(max_seq_len, 100))
                rnn_layer = masking(input)

                enc = self.setup_encoder("lstm")
                rnn_layer = enc(rnn_layer)

                attention = self.setup_attention("lstm")
                rnn_layer, att = attention(rnn_layer)

                rnn_inputs.append(rnn_layer)
                # self.att_scores.append(att)

            # join inputs
            if len(algonet_inputs) > 1:
                join_op = self.setup_join_inputs()
                rnn_layer = join_op(rnn_inputs)
            else:
                rnn_layer = rnn_inputs[0]

            layer = Concatenate()([layer, rnn_layer])

        #######################################

        classifier = self.setup_classifier(layer, self.params["classifier"])
        outputs    = [classifier]
        loss       = ["binary_crossentropy"]

        if "loss" in self.params and self.params["loss"] == "focal_loss":
            loss[0] = AlgoLabel.focal_loss

        logging.info("Outputs", outputs)
        logging.info("Loss", loss)

        model = Model(inputs=self.inputs,
                      outputs=outputs)

        model.compile(loss=loss,
                      optimizer=self.params["optimizer"],
                      metrics=self.params["metrics"])

        self.model = model
        print(model.summary())

        return model

    def setup_embedding(self, input, emb_input_type, matrix_name):

        emb_scenario = input["scenario"]
        emb_type     = self.args["embeddings"][emb_scenario]["emb_type"]
        emb_input    = self.args["embeddings"][emb_scenario]["input_type"]
        emb_params   = self.args["embeddings"]["framework"][emb_type]
        encoder      = self.args["encoders"][input["encoder"]]
        max_seq_len  = encoder["max_seq_len_{}".format(emb_input_type)]

        if emb_scenario not in self.embbedings:
            if emb_type != "word2vec":
                raise NotImplementedError("Unrecognized embedding type!")

            load_word2vec(self.args, emb_input_type)

            vocab = self.args["{}_vocab_size".format(matrix_name)]
            emb_matrix = self.args["{}_matrix".format(matrix_name)]

            embedding = Embedding(input_dim=vocab,
                                  output_dim=emb_params["size"],
                                  embeddings_initializer=Constant(emb_matrix),
                                  input_length=max_seq_len)

            self.embbedings[emb_scenario] = embedding

        return self.embbedings[emb_scenario]

    def compute_matrix_name(self, input, emb_input):

        matrix_name = "w2v_"
        if emb_input == "text":
            emb_input_type = emb_input
            matrix_name += "text"
        elif emb_input == "tokens" or emb_input == "symbs":
            emb_input_type = "code_{}".format(input["field"])
            matrix_name += "code" + "_" + input["field"]
        else:
            raise NotImplementedError("Unrecognized input type")

        return emb_input_type, matrix_name

    def setup_input_layer(self, input, layer_name=None):

        emb_scenario = input["scenario"]

        if emb_scenario == "safe":
            return self.setup_safe_input_layer(input)
        elif emb_scenario == "code2vec":
            return self.setup_ast_input_layer(input)
        elif emb_scenario == "w2v_cnn_text":
            return self.setup_cnn_input_layer(input)

        encoder_name = input["encoder"]
        encoder      = self.args["encoders"][input["encoder"]]

        emb_type     = self.args["embeddings"][emb_scenario]["emb_type"]
        emb_input    = self.args["embeddings"][emb_scenario]["input_type"]
        emb_params   = self.args["embeddings"]["framework"][emb_type]
        max_seq_len  = encoder["max_seq_len_{}".format(emb_input)]

        emb_input_type, matrix_name = self.compute_matrix_name(input, emb_input)

        if not layer_name:
            name  = "{}_encoder".format(input["field"])
        else:
            name = layer_name

        layer = Input(shape=(max_seq_len,), name=name)
        self.inputs.append(layer)

        emb     = self.setup_embedding(input, emb_input_type, matrix_name)
        layer   = emb(layer)

        pad_idx = len(self.args["{}_matrix".format(matrix_name)]) - 1
        masking = Masking(mask_value=pad_idx,
                          input_shape=(max_seq_len, emb_params["size"]))
        layer   = masking(layer)

        enc     = self.setup_encoder(encoder_name)
        layer   = enc(layer)

        if encoder["attention"]:
            print(encoder_name)
            print(layer)
            attention = self.setup_attention(encoder_name)
            print(attention)
            layer, att = attention(layer)
            self.att_scores.append(att)

        return layer

    def setup_join_inputs(self):

        join_op = self.params["encoders"]["join_operation"]

        if join_op != "default":
            raise NotImplementedError("Unrecognized join operation")

        return Concatenate()

    # def join_ast_inputs(self):
    #
    #     join_op     = self.params["encoders"]["join_operation"]
    #     join_params = self.params["join"]["ast"]
    #
    #     context_embedded = Concatenate()(self.inputs[:3])
    #     context_embedded = Dropout(0.1)(context_embedded)
    #     context_after_dense = TimeDistributed(
    #         Dense(join_params["ast"],
    #               use_bias=False,
    #               activation='tanh'))(context_embedded)
    #
    #     # code_vectors, attention_weights = Code2VecAttentionLayer(name="ast_attention")(
    #     #     [context_after_dense, context_valid_mask]
    #     # )

    def setup_classifier(self, layer, cls_type):

        if cls_type not in self.args["classifier"]:
            raise NotImplementedError("Unrecognized classifier layer")

        cls_args = self.args["classifier"][cls_type]
        for dense_size in cls_args["dense"]:
            if "dropout" in cls_args:
                layer = Dropout(cls_args["dropout"])(layer)
            layer = Dense(dense_size,
                          activation=cls_args["activation"],
                          kernel_regularizer=l2(cls_args["regularizer"]),
                          bias_regularizer=l2(cls_args["regularizer"]),
                          name=cls_type)(layer)

        return layer

    def build_model(self, extract=False, extra_supervision=False):

        inputs = self.params["encoders"]["inputs"]

        print("MODEL NAME", self.model_name)
        if self.model_name == "EMNLP":
            return self.build_emnlp_model()

        # setup input layers
        for input in inputs:
            layer = self.setup_input_layer(input)
            self.input_layers.append(layer)

        # join inputs
        if len(inputs) > 1:
            join_op    = self.setup_join_inputs()
            classifier = join_op(self.input_layers)
        else:
            classifier = self.input_layers[0]

        # pass to classifier
        prev_classifier = classifier

        for cls_type in self.params["classifiers"]:
            prev_classifier = classifier
            classifier = self.setup_classifier(classifier, cls_type)

        outputs = [classifier] + self.att_scores
        loss    = ["binary_crossentropy"] + [None] * len(self.att_scores)

        if "loss" in self.params and self.params["loss"] == "focal_loss":
            loss[0] = tfa.losses.sigmoid_focal_crossentropy

        if extract:
            outputs.append(prev_classifier)
            loss.append(None)
        elif extra_supervision:
            outputs.append(prev_classifier)
            loss.append("binary_crossentropy")

        print("Inputs", self.inputs)
        print("Outputs", outputs)
        print("Loss", loss)

        model = Model(inputs=self.inputs,
                      outputs=outputs)

        model.compile(loss=loss,
                      optimizer=self.params["optimizer"],
                      metrics=self.params["metrics"])

        self.model = model
        print(model.summary())

        return model

    def setup_callbacks(self):

        callbacks = self.args["train"]["callbacks"]

        for callback in callbacks:

            params = self.args["callbacks"][callback]

            if callback == "early_stopping":
                cb = EarlyStopping(
                    monitor=params["monitor"],
                    mode=params["mode"],
                    patience=params["patience"],
                    verbose=params["verbose"],
                    min_delta=params["min_delta"]
                )
            elif callback == "checkpoint":
                path = "./data/models/{}/models/epoch_".format(self.model_name)

                if self.load_epoch:
                    path += "{}_".format(self.load_epoch)

                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

                cb = ModelCheckpoint(
                    path + "{epoch:02d}.h5",
                    monitor=params["monitor"],
                    mode=params["mode"],
                    save_best_only=params["save_best_only"],
                    verbose=params["verbose"],
                )

            else:
                print("Unrecognized callback!")
                exit(1)
                return

            self.callbacks.append(cb)

    def setup_weights(self):

        if self.load_epoch:
            path  = "./data/models/{}/models/epoch_{:02d}.h5".format(self.model_name, self.load_epoch)
            print("Load weights from epoch {} - {}".format(self.load_epoch, path))
            self.model.load_weights(path)
        else:
            self.load_epoch = 0

    def train(self):

        X_train, Y_train = self.args["X_train"], self.args["Y_train"]
        X_dev, Y_dev     = self.args["X_dev"], self.args["Y_dev"]
        self.setup_callbacks()

        if self.args["train"]["extra_supervision"]:
            Y_train_extra = self.args["Y_train_source_emb"]
            Y_dev_extra   = self.args["Y_dev_source_emb"]

            Y_train = {
                "cherry": Y_train,
                "problem_emb": Y_train_extra
            }

            Y_dev = {
                "cherry": Y_dev,
                "problem_emb": Y_dev_extra
            }
            print("Extra supervision!")

        if self.args["train"]["debug"]:
            for x in range(len(X_train)):
                X_train[x] = X_train[x][:10]
                X_dev[x] = X_dev[x][:10]
            Y_train = Y_train[:10]
            Y_dev = Y_dev[:10]

        history = self.model.fit(X_train, Y_train,
                                 validation_data=(X_dev, Y_dev),
                                 epochs=self.args["train"]["num_epochs"] - self.load_epoch,
                                 batch_size=self.args["train"]["batch_size"],
                                 callbacks=self.callbacks,
                                 shuffle=True,
                                 verbose=2)

        model_path = "./data/models/{}/models".format(self.model_name)
        path = "{}/epoch_{}.h5".format(model_path, self.args["train"]["num_epochs"])
        self.save_model(path)

    def save_model(self, path):

        root = "/".join(path.split("/")[:-1])
        if not os.path.exists(path):
            os.makedirs(root, exist_ok=True)
        self.model.save_weights(path)

    def log_results_header(self, f, idx, test, pred, sample):
        f.write("\n\nSample {}\n".format(idx))
        f.write("\n\t\tCorrect\tPredicted\n")
        for idx, label in enumerate(self.args["split"]["labels"]):
            f.write("\n{}:\t{}\t{}\n".format(
                label, test[idx], pred[idx]
            ))
        f.write("\n")
        f.write(sample["url"] + "\n")

    def validate_results_text(self, test_data, Y_test, Y_pred):

        difficulty_distro = {
            "Easy": ([], []),
            "Medium": ([], []),
            "Hard": ([], [])
        }

        with open("./logs/text_results_analysis.log", "w") as f:

            for idx, (y_test, y_pred) in enumerate(zip(Y_test, Y_pred)):
                sample = test_data[idx]
                difficulty = sample["difficulty_class"]

                test, pred = difficulty_distro[difficulty]
                test.append(y_test)
                pred.append(y_pred)

                if difficulty == "Hard" and "graphs" in sample["tags"]:
                    self.log_results_header(f, idx, y_test, y_pred, sample)
                    f.write("{}\n\n{}\n\n{}\n\n".format(
                            sample["statement"],
                            sample["input"],
                            sample["output"]))

        for diff in difficulty_distro:
            self.compute_results(np.array(difficulty_distro[diff][0]),
                                 np.array(difficulty_distro[diff][1]),
                                 diff)

    def validate_results_code(self, test_data, Y_test, Y_pred):

        problems = defaultdict(list)

        cls_threshold = self.args["test"]["threshold"]
        Y_pred[Y_pred < cls_threshold] = 0
        Y_pred[Y_pred >= cls_threshold] = 1

        for idx, sample in enumerate(test_data):

            problem_index = sample["index"].split("_")[1]
            problems[problem_index].append(Y_pred[idx])

        a_sum = 0
        for idx, problem in enumerate(problems):
            # problem[problem_index] = np.array(problem[problem_index])
            a = alpha(problems[idx])
            a_sum += a
            print("Alpha", idx, a)

        print("Average Alpha", a_sum / len(problems))

        with open("./logs/code_results_analysis.log", "w") as f:
            for idx, (y_test, y_pred) in enumerate(zip(Y_test, Y_pred)):
                sample = test_data[idx]
                self.log_results_header(f, idx, y_test, y_pred)
                f.write("\nSource:\n")
                f.write(sample["code"])
                f.write("\n")
                sample["predictions"] = y_pred

    def compute_results(self, test_res, pred_res, ds):

        cls_threshold = self.args["test"]["threshold"]
        targets       = self.args["split"]["labels"]

        print("Computing results for the {} dataset!".format(ds))

        print(test_res.shape)
        print(pred_res.shape)

        auc = roc_auc_score(test_res, pred_res)
        print("ROC-AUC: {}".format(auc))

        pred_res[pred_res < cls_threshold] = 0
        pred_res[pred_res >= cls_threshold] = 1

        hamming = hamming_loss(test_res, pred_res)
        print("Hamming Loss {}".format(hamming))

        result = classification_report(test_res, pred_res, target_names=targets)
        print(result)

        save_result_log(self.args,
                        result, {"AUC": auc, "Hamming": hamming}, ds=ds)

        print("Model: {}".format(self.model_name.upper()))

    def load_test_data(self, input_type, split="test"):
        if input_type == "text":
            return load_dataset("./data/datasets/split/{}/{}.json".format(input_type, split))
        else:
            return load_dataset("./data/datasets/split/{}/{}_all.json".format(input_type, split))

    def test(self, split="test"):

        X_test, Y_test   = self.args["X_test"], self.args["Y_test"]

        if not self.load_epoch:
            self.load_epoch = self.args["train"]["num_epochs"]
            self.setup_weights()

        result     = self.model.predict(X_test)
        input_type = self.args["models"][self.model_name]["encoders"]["input_type"]

        if len(self.att_scores) == 0:
            Y_pred = result
        else:
            Y_pred = result[0]

        if input_type == "text" or input_type == "both":
            test_data = self.load_test_data("text")
            self.validate_results_text(test_data, Y_test, Y_pred)

        if input_type == "code":
            test_data = self.load_test_data("code")
            self.validate_results_code(test_data, Y_test, Y_pred)

        self.compute_results(Y_test, Y_pred, "full")

        # dump_dataset("./data/datasets/split/{}/{}_{}_{}.json".format(input_type,
        #                                                              split,
        #                                                              self.model_name,
        #                                                              self.load_epoch),
        #              test_data)

    def extract_embeddings(self, split):

        X_test, Y_test   = self.args["X_{}".format(split)], self.args["Y_{}".format(split)]

        if not self.load_epoch:
            self.load_epoch = self.args["train"]["num_epochs"]
            self.setup_weights()

        result     = self.model.predict(X_test)
        input_type = self.args["models"][self.model_name]["encoders"]["input_type"]
        embedding  = result[-1]

        print("Embeddings", embedding.shape)

        # if len(self.att_scores) == 0:
        #     Y_pred = result
        # else:
        #     Y_pred = result[0]
        # self.compute_results(Y_test, Y_pred, "full_{}".format(split))

        test_data = self.load_test_data(input_type, split=split)

        problem_emb = defaultdict(list)
        label_emb   = defaultdict(list)

        for idx, sample in enumerate(test_data):

            problem_index = sample["index"].split("_")[1]
            problem_emb[problem_index].append(embedding[idx])

            for tag in sample["tags"]:
                label_emb[tag].append(embedding[idx])

        result = {
            "problem_emb": {},
            "label_emb": {}
        }

        for problem in problem_emb:
            result["problem_emb"][problem] = list(map(float, list(np.average(problem_emb[problem], axis=0))))

        for label in label_emb:
            result["label_emb"][label] = list(map(float, list(np.average(label_emb[label], axis=0))))

        dump_dataset("./data/embeddings/source_emb_{}.json".format(split), result)








