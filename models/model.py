import logging
import os

from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, roc_auc_score, hamming_loss

from util import generic_serialize, generic_deserialize, ensure_path
from pathlib import Path

from logs.utils import save_result_log
from util import tf_gpu_housekeeping


class Model(object):

    def __init__(self, args):

        self.args  = args
        self.name  = args["model"]

        self.model_name = None
        self.model      = None

        self.inputs     = []
        self.att_scores = []

        self.callbacks  = []
        self.load_epoch = self.args["load_weights_from_epoch"]

        tf_gpu_housekeeping()

    def prepare(self, X):
        '''
            Builds a specialized dataset for this particular model type.
            :param X: dictionary with the raw data
            :return: preprocessed dataset type
        '''
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def get_model_path(self):
        path = Path.cwd() / "data" / "models" / self.model_name
        ensure_path(path)
        return path

    def load_weights(self):

        if not self.model:
            raise Exception("Model not initialized!")

        if self.load_epoch:
            path = self.get_model_path()
            path /= "epoch_{:02d}.h5".format(self.load_epoch)

            if not os.path.exists(path):
                raise Exception("Pretrained model {} not available!".format(path))
            self.model.load_weights(path)
        else:
            self.load_epoch = 0

    def fit(self, train_data, dev_data):

        X_train, Y_train = train_data
        X_dev, Y_dev     = dev_data

        self._setup_callbacks()
        self.load_weights()

        history = self.model.fit(X_train, Y_train,
                                 validation_data=(X_dev, Y_dev),
                                 epochs=self.args["train"]["num_epochs"] - self.load_epoch,
                                 batch_size=self.args["train"]["batch_size"],
                                 callbacks=self.callbacks,
                                 shuffle=True,
                                 verbose=2)

        model_path = self.get_model_path() / "final.h5"
        self.serialize(model_path)

    def score(self, test_data):

        X_test, Y_test = test_data

        if not self.load_epoch:
            self.load_epoch = self.args["train"]["num_epochs"]
        self.load_weights()

        result = self.model.predict(X_test)

        if len(self.att_scores) == 0:
            Y_pred = result
        else:
            Y_pred = result[0]

        self.compute_results(Y_test, Y_pred)

    def compute_results(self, test_res, pred_res, ds="full"):

        cls_threshold = self.args["test"]["threshold"]
        targets       = self.args["split"]["labels"]

        logging.info("Computing results using {} on the {} dataset!".format(
            self.model_name, ds)
        )

        auc = roc_auc_score(test_res, pred_res)
        logging.info("ROC-AUC: {}".format(auc))

        pred_res[pred_res < cls_threshold]  = 0
        pred_res[pred_res >= cls_threshold] = 1

        hamming = hamming_loss(test_res, pred_res)
        logging.info("Hamming Loss {}".format(hamming))

        cls_report = classification_report(test_res, pred_res, target_names=targets)
        logging.info(cls_report)

        metrics_report = {"AUC": auc, "Hamming": hamming}
        save_result_log(self.args, cls_report, metrics_report, ds=ds)

    def predict(self, x):
        raise NotImplementedError()

    def get_config(self):
        return self.args

    @staticmethod
    def setup_early_stopping(params):

        return EarlyStopping(
                    monitor=params["monitor"],
                    mode=params["mode"],
                    patience=params["patience"],
                    verbose=params["verbose"],
                    min_delta=params["min_delta"]
                )

    def setup_model_checkpoint(self, params):

        path = self.get_model_path()
        if self.load_epoch:
            path /= "epoch_{}_".format(self.load_epoch)
        else:
            path /= "epoch_"

        cb = ModelCheckpoint(
            str(path) + "{epoch:02d}.h5",
            monitor=params["monitor"],
            mode=params["mode"],
            save_best_only=params["save_best_only"],
            verbose=params["verbose"],
        )

        return cb

    def _setup_callbacks(self):

        callbacks = self.args["train"]["callbacks"]

        for callback in callbacks:

            params = self.args["callbacks"][callback]

            if callback == "early_stopping":
                cb = self.setup_early_stopping(params)
            elif callback == "checkpoint":
                cb = self.setup_model_checkpoint(params)
            else:
                raise Exception("Unsupported callback option! {}".format(callback))

            self.callbacks.append(cb)

    def serialize(self, path: str):
        generic_serialize(self, str(path))

    def deserialize(self, path: str):
        return generic_deserialize(str(path))

    def log_results_header(self, f, idx, test, pred, sample):
        f.write("\n\nSample {}\n".format(idx))
        f.write("\n\t\tCorrect\tPredicted\n")
        for idx, label in enumerate(self.args["split"]["labels"]):
            f.write("\n{}:\t{}\t{}\n".format(
                label, test[idx], pred[idx]
            ))
        f.write("\n")
        f.write(sample["url"] + "\n")
