from util import fcall, load_dataset, tf_gpu_housekeeping
import numpy as np

from sklearn.metrics import classification_report, roc_auc_score, hamming_loss
from models.backup.algonet import build_algonet, build_algohan
from models.backup.bert import build_bert_model, build_3bert_model
from logs.utils import save_result_log

from tensorflow import convert_to_tensor, int32
from keras.callbacks import TensorBoard
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint

import logging

from models.algolabel import AlgoLabel


@fcall
def evaluate_model(args, model, label=""):
    X_test = args["X_test"]
    Y_test = np.array(args["Y_test"])

    if args["current_model"] == "AlgoNet":
        X_test = [X_test[0], X_test[1], X_test[2]]
    elif args["current_model"] == "AlgoHan":
        X_test = np.array(args["X_test"])
    elif args["current_model"] == "BertAlgoNet":
        X_test = [np.array(X_test[source]) for source in ["ids", "masks", "segments"]]
    elif args["current_model"] == "3BertAlgoNet":
        X_test = []
        for source in ["statement", "input", "output"]:
            for input_type in ["ids", "masks", "segments"]:
                X_test.append(np.array(args["X_test"][source][input_type]))

    model_path = "./data/models/{}".format(args["current_model"])
    model.load_weights(model_path + "/epoch_{}.h5".format(args["num_epochs"]))

    model_params = args["models"][args["current_model"]]
    if "word_attention" in model_params and model_params["word_attention"]:
        Y_pred, statement_att, input_att, output_att = model.predict(X_test)
    elif "sent_attention" in model_params and model_params["sent_attention"]:
        Y_pred, sent_att = model.predict(X_test)
    else:
        Y_pred = model.predict(X_test)

    targets = args["split_dataset"]["targets"]

    # test_data = load_dataset("./data/test.json")
    # for idx, (y_test, y_pred) in enumerate(zip(Y_test, Y_pred)):
    #
    #     if y_pred[2] < 0.5 and y_test[2] == 1:
    #         print("[graph classification error]")
    #         print("Statement\n{}\n**Input**\n{}\n**Output**\n{}\n**Gold Tags**\n{}\nUrl{}".format(test_data[idx]["statement"],
    #                                                                            test_data[idx]["input"],
    #                                                                            test_data[idx]["output"],
    #                                                                            test_data[idx]["tags"],
    #                                                                            test_data[idx]["url"]))
    #         print("           {}\n".format(targets))
    #         print("Prediction {}\nFinal {}\nGold {}\n".format(y_pred, y_pred >= 0.4, y_test))
    #         print("\n")

    difficulty_distro = {
        "Easy": ([], []),
        "Medium": ([], []),
        "Hard": ([], [])
    }

    test_data = load_dataset("./data/datasets/test.json")
    for idx, (y_test, y_pred) in enumerate(zip(Y_test, Y_pred)):
        sample     = test_data[idx]
        difficulty = sample["difficulty"]

        test, pred = difficulty_distro[difficulty]
        test.append(y_test)
        pred.append(y_pred)

    def compute_results(test_res, pred_res, ds):

        print("Computing results for the {} dataset!".format(ds))

        auc = roc_auc_score(test_res, pred_res)
        print("ROC-AUC: {}".format(auc))

        pred_res[pred_res < 0.4]  = 0
        pred_res[pred_res >= 0.4] = 1

        hamming = hamming_loss(test_res, pred_res)
        print("Hamming Loss {}".format(hamming))

        result = classification_report(test_res, pred_res, target_names=targets)
        print(result)

        save_result_log(args,
                        result, {
                            "AUC": auc,
                            "Hamming": hamming,
                        },
                        ds=ds
                        )

    compute_results(Y_test, Y_pred, "full")

    for diff in difficulty_distro:
        compute_results(np.array(difficulty_distro[diff][0]),
                        np.array(difficulty_distro[diff][1]),
                        diff)


@fcall
def run_model(args, model, label=""):
    model_path = "./data/models/{}".format(args["current_model"])

    if args["load_weights_from_epoch"]:
        print("load weights from epoch {}".format(args["load_weights_from_epoch"]))
        model.load_weights(model_path + "/epoch_{}.h5".format(args["load_weights_from_epoch"]))
    else:
        args["load_weights_from_epoch"] = 0

    X_train, Y_train = args["X_train"], args["Y_train"]
    X_dev, Y_dev     = args["X_dev"], args["Y_dev"]
    X_test, Y_test   = args["X_test"], args["Y_test"]

    if args["current_model"] == "AlgoHan":
        X_train = np.array(X_train)
        X_dev   = np.array(X_dev)
        X_test  = np.array(X_test)
    elif args["current_model"] == "AlgoNet":
        X_train = [X_train[0], X_train[1], X_train[2]]
        X_dev   = [X_dev[0], X_dev[1], X_dev[2]]
        X_test  = [X_test[0], X_test[1], X_test[2]]
    elif args["current_model"] == "BertAlgoNet":
        X_train = [np.array(X_train[source]) for source in ["ids", "masks", "segments"]]
        X_dev   = [np.array(X_dev[source]) for source in ["ids", "masks", "segments"]]
        # X_test  = [np.array(X_test[source])  for source in ["ids", "masks", "segments"]]

        X_train = [
            convert_to_tensor(X_train[0], dtype=int32, name="input_word_ids"),
            convert_to_tensor(X_train[1], dtype=int32, name="input_mask"),
            convert_to_tensor(X_train[2], dtype=int32, name="input_type_ids")
        ]

        X_dev = [
            convert_to_tensor(X_dev[0], dtype=int32, name="input_word_ids"),
            convert_to_tensor(X_dev[1], dtype=int32, name="input_mask"),
            convert_to_tensor(X_dev[2], dtype=int32, name="input_type_ids")
        ]
    elif args["current_model"] == "3BertAlgoNet":

        train, dev     = args["X_train"], args["X_dev"]
        X_train, X_dev = [], []

        for source in ["statement", "input", "output"]:
            for input_type in ["ids", "masks", "segments"]:
                X_train.append(np.array(train[source][input_type]))
                X_dev.append(np.array(dev[source][input_type]))

    else:
        print("[run_model] Unsupported model! {}".format(args["current_model"]))
        exit(1)

    es = EarlyStopping(monitor='val_dense_2_accuracy',
                       mode='max',
                       patience=args["patience"],
                       verbose=1,
                       min_delta=0.001)

    mc = ModelCheckpoint(model_path + "/epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5",
                         monitor='val_dense_2_accuracy',
                         mode='max',
                         save_best_only=True,
                         verbose=1)

    tb = TensorBoard(log_dir="./logs/tensorboard",
                     histogram_freq=0,
                     update_freq='epoch',
                     )

    history = model.fit(X_train, Y_train,
                        validation_data=(X_dev, Y_dev),
                        epochs=args["num_epochs"] - args["load_weights_from_epoch"],
                        batch_size=args["batch_size"],  # 32
                        callbacks=[es, tb],
                        shuffle=True,
                        verbose=2)

    model.save_weights(model_path + "/epoch_{}.h5".format(args["num_epochs"]))
    evaluate_model(args, model, label)

    scores = model.evaluate(X_test, Y_test, verbose=1)
    logging.info(scores)


def build_model(args):
    if args["current_model"] == "AlgoNet":
        return build_algonet(args)
    elif args["current_model"] == "AlgoHan":
        return build_algohan(args)
    elif args["current_model"] == "BertAlgoNet":
        return build_bert_model(args)
    elif args["current_model"] == "3BertAlgoNet":
        return build_3bert_model(args)

    print("Unrecognized model type! {}".format(args["current_model"]))
    exit(1)


@fcall
def setup_model(args, extract=False, extra_supervision=False):
    solution = AlgoLabel(args)
    solution.build_model(extract=extract, extra_supervision=extra_supervision)
    solution.setup_weights()
    return solution


@fcall
def load_input(args, load_tokens=False, load_extra_supervision=False):
    model_name = args["model"]
    print("Load tokens:", load_tokens)
    print("Load extra supervision:", load_extra_supervision)

    for split in ["train", "dev", "test"]:

        args["X_{}".format(split)] = load_dataset(
            "./data/models/{}/data/X_{}.json".format(model_name, split))

        args["Y_{}".format(split)] = np.array(load_dataset(
            "./data/models/{}/data/Y_{}.json".format(model_name, split)))

        if load_tokens:
            args["X_toks_{}".format(split)] = load_dataset(
                "./data/models/{}/data/X_toks_{}.json".format(model_name, split))

        if load_extra_supervision:
            args["Y_{}_source_emb".format(split)] = np.array(load_dataset(
                "./data/models/{}/data/Y_{}_source_emb.json".format(model_name, split)
            ))


@fcall
def train(args):
    tf_gpu_housekeeping()
    model = setup_model(args, extract=False, extra_supervision=args["train"]["extra_supervision"])
    load_input(args, load_tokens=False, load_extra_supervision=args["train"]["extra_supervision"])
    model.train()
    model.test()


@fcall
def test(args):
    tf_gpu_housekeeping()
    model = setup_model(args)
    load_input(args)
    model.test()


@fcall
def extract_embeddings(args):
    tf_gpu_housekeeping()
    model = setup_model(args, extract=True)
    load_input(args)

    for split in ["train", "dev", "test"]:
        model.extract_embeddings(split)
