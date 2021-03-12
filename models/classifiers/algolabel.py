from keras.models import Sequential, Model as KerasModel
from keras.layers import Masking, Concatenate, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Conv1D, ReLU, GRU
from keras.layers import MaxPool1D, Permute, Flatten
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.regularizers import l2

from models.layers.attention import CustomAttention
from models.model import Model
from models.embeddings.word2vec import Word2VecEmbedding

from pathlib import Path
from util import ensure_path
import logging


class AlgoLabel(Model):

    def __init__(self, args):
        super().__init__(args)

        self.encoders   = {}
        self.embeddings = {}
        self.attention  = {}

    def _setup_encoder(self, encoder_name):

        if encoder_name not in self.args["encoders"]:
            raise NotImplementedError("Unsupported encoder type {}"
                                      .format(encoder_name))

        # Check cached encoder layer
        if encoder_name in self.encoders:
            return self.encoders[encoder_name]

        encoder = self.args["encoders"][encoder_name]

        if encoder["regularizer"]:
            regularizer = l2(encoder["regularizer"])
        else:
            regularizer = None

        if "lstm" in encoder_name:
            rnn = LSTM
        elif "gru" in encoder_name:
            rnn = GRU
        else:
            raise NotImplementedError("Unsupported encoder type {}".format(encoder_name))

        layer = Bidirectional(rnn(
            units=encoder["hidden_size"],
            dropout=encoder["dropout"],
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            return_sequences=encoder["attention"]
        ))

        self.encoders[encoder_name] = layer
        return layer

    def _setup_attention(self, encoder_params, layer, encoder_name):

        if not encoder_params["attention"]:
            return layer

        if encoder_name not in self.attention:
            self.attention[encoder_name] = CustomAttention()

        layer, att = self.attention[encoder_name](layer)
        self.att_scores.append(att)
        return layer

    def _setup_code2vec_input_layer(self, input_params):

        encoder_name   = input_params["encoder"]
        encoder_params = self.args["encoders"][encoder_name]
        max_seq_len    = encoder_params["max_seq_len"]

        start_token_input = Input(shape=(max_seq_len,),
                                  name="ast_path_start", )
        path_input        = Input(shape=(max_seq_len,),
                                  name="ast_path_index", )
        end_token_input   = Input(shape=(max_seq_len,),
                                  name="ast_path_end", )

        self.inputs += [start_token_input, path_input, end_token_input]

        token_embedding_shared_layers = Embedding(
            input_dim=encoder_params["token_vocab_size"],
            output_dim=encoder_params["token_emb_size"],
            name="token_embedding",
            mask_zero=True,
            input_length=max_seq_len
        )

        path_embedding = Embedding(
            input_dim=encoder_params["path_vocab_size"],
            output_dim=encoder_params["path_emb_size"],
            name="path_embedding",
            mask_zero=True
        )

        path_start_emb = token_embedding_shared_layers(start_token_input)
        path_end_emb   = token_embedding_shared_layers(end_token_input)
        path_emb       = path_embedding(path_input)

        context_emb = Concatenate(name="ast_context")([
            path_start_emb, path_emb, path_end_emb
        ])

        context_emb         = Dropout(encoder_params["dropout"])(context_emb)
        context_dense       = Dense(2 * encoder_params["token_emb_size"]
                                    + encoder_params["path_emb_size"],
                                    use_bias=False,
                                    activation="tanh")
        context_after_dense = TimeDistributed(context_dense)(context_emb)
        layer               = self._setup_attention(encoder_params, context_after_dense, encoder_name)
        return layer

    def _setup_safe_input_layer(self, input_params):

        encoder_name   = input_params["encoder"]
        encoder_params = self.args["encoders"][encoder_name]
        max_seq_len    = encoder_params["max_seq_len"]
        input_size     = encoder_params["input_size"]

        layer = Input(shape=(max_seq_len, input_size),
                      name="{}_encoder".format(input_params["field"]))
        self.inputs.append(layer)

        masking = Masking(mask_value=0,
                          input_shape=(max_seq_len, input_size))

        layer   = masking(layer)
        layer   = self._setup_encoder(encoder_name)(layer)
        layer   = self._setup_attention(encoder_params, layer, encoder_name)
        return layer

    def setup_embedding(self, input_params):

        scenario_name = input_params["scenario"]
        emb_scenario  = self.args["features"]["scenarios"][scenario_name]
        emb_input     = emb_scenario["input"]
        emb_field     = emb_scenario["field"]

        if scenario_name not in self.embeddings:

            if "word2vec" not in scenario_name:
                raise NotImplementedError("Unsupported embedding type {}".format(scenario_name))

            embedding = Word2VecEmbedding(self.args,
                                          input_type=emb_input,
                                          input_field=emb_field)
            embedding.load_weights()
            self.embeddings[scenario_name] = embedding

        return self.embeddings[scenario_name]

    def _setup_generic_input_layer(self, input_params):

        encoder_name   = input_params["encoder"]
        encoder_params = self.args["encoders"][encoder_name]
        scenario       = self.args["features"]["scenarios"][input_params["scenario"]]
        emb_input      = scenario["input"]
        emb_field      = scenario["field"]
        max_seq_len    = encoder_params["max_seq_len_{}".format(emb_input)]

        layer         = Input(shape=(max_seq_len,), name="{}_encoder".format(emb_field))
        self.inputs.append(layer)

        embedding = self.setup_embedding(input_params)
        layer     = embedding.apply_to_input_layer(layer, max_seq_len, apply_mask=True)

        encoder   = self._setup_encoder(encoder_name)
        layer     = encoder(layer)

        layer     = self._setup_attention(encoder_params, layer, encoder_name)

        return layer

    def _setup_input_layer(self, input_params):

        emb_scenario = input_params["scenario"]

        if "safe" in emb_scenario:
            return self._setup_safe_input_layer(input_params)
        elif "code2vec" in emb_scenario:
            return self._setup_code2vec_input_layer(input_params)
        else:
            return self._setup_generic_input_layer(input_params)

    def _setup_join_operation(self, op_type):

        if op_type != "default":
            raise NotImplementedError("Unsupported join operation {}".format(op_type))
        return Concatenate()

    def _setup_classifier(self, layer, cls_type):

        if cls_type not in self.args["classifiers"]:
            raise NotImplementedError("Unsupported classifier type {}".format(cls_type))

        params = self.args["classifiers"][cls_type]

        for idx, output_size in enumerate(params["dense"]):
            if "dropout" in params:
                layer = Dropout(params["dropout"])(layer)

            layer = Dense(output_size,
                          activation=params["activation"],
                          kernel_regularizer=l2(params["regularizer"]),
                          bias_regularizer=l2(params["regularizer"]),
                          name="{}_{}".format(cls_type, idx)
                          )(layer)
        return layer

    def build_model(self, model_name=None, verbose=True, extract_representation=False):

        if not model_name:
            self.model_name = self.args["model"]
        else:
            self.model_name = model_name

        if verbose:
            logging.info("Building model {}".format(self.model_name))

        params = self.args["models"][self.model_name]
        inputs = params["encoders"]["inputs"]

        if len(inputs) == 0:
            raise Exception("No input defined for model {}".format(self.model_name))

        base_layers = []
        for input in inputs:
            layer = self._setup_input_layer(input)
            base_layers.append(layer)

        if len(base_layers) > 1:
            join_op = self._setup_join_operation(params["encoders"]["join_operation"])
            classifier = join_op(base_layers)
        else:
            classifier = base_layers[0]

        prev_cls = classifier

        if "classifiers" in params:
            for cls_type in params["classifiers"]:
                prev_cls   = classifier
                classifier = self._setup_classifier(classifier, cls_type)

        outputs = [classifier] + self.att_scores
        loss    = [params["loss"]] + [None] * len(self.att_scores)

        if extract_representation:
            outputs.append(prev_cls)
            loss.append(None)

        model = KerasModel(name=self.model_name, inputs=self.inputs, outputs=outputs)
        model.compile(loss=loss,
                      optimizer=params["optimizer"],
                      metrics=params["metrics"])

        self.model = model
        if verbose:
            logging.info(model.summary())

        return model

    def serialize(self, path: Path):
        ensure_path(path.parent)
        self.model.save_weights(str(path))

    def deserialize(self, path: str):
        self.model.load_weights(str(path))
