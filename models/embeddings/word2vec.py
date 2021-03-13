from gensim.models import Word2Vec
from models.embeddings.embedding import Embedding
from util import fcall, load_dataset
import numpy as np
import logging
from pathlib import Path
from keras.layers.embeddings import Embedding as KerasEmb
from keras.initializers import Constant
from keras.layers import Masking


class Word2VecEmbedding(Embedding):

    def __init__(self, args, output_size=None, input_type="text", input_field=None):
        '''
        :param args:
        :param output_size:
        :param input_type: One of {"text", "code"}
        '''
        self.params = args["features"]["types"]["word2vec"]
        output_size = self.params["size"] if not output_size else output_size
        self.input_type = input_type

        super(Word2VecEmbedding, self).__init__(args, output_size)

        self.matrix     = None
        self.vocab_size = None
        self.emb_path  /= "w2v_{}_{}.emb".format(input_type, output_size)
        self.model      = None

        self.input_field = input_field

        # Text inputs share the same embeddings
        if self.input_field == "text":
            text_fields       = self.args["prepare"]["text"]["fields"]
            self.input_fields = [field for field in text_fields
                                 if text_fields[field]]
        if not input_field:
            raise NotImplementedError("Word2Vec embedding - input_field not defined")

        self.weights_loaded = False

    @fcall
    def load_weights(self, path=None):

        if not path:
            path = self.emb_path

        # Load pretrained Word2Vec model
        self.model      = Word2Vec.load(str(path))

        # Attach special 'unk' and 'pad' vectors to embedding matrix
        emb_matrix      = np.array(self.model.wv.vectors)
        unk_vector      = np.mean(emb_matrix, axis = 0)
        self.unk_idx    = len(emb_matrix)
        self.pad_idx    = len(emb_matrix) + 1
        self.matrix     = np.append(emb_matrix, [unk_vector, self.pad_vector], axis=0)
        self.vocab_size = self.matrix.shape[0]

        self.weights_loaded = True
        return self

    def get_token_id(self, token):
        if token in self.model:
            return int(self.model.wv.vocab.get(token).index)
        return self.unk_idx

    @fcall
    def pretrain(self, X):

        if self.input_field == "text":
            inputs = []
            for sample in X:
                for field in self.input_fields:
                    if not field in sample:
                        print(sample)
                        exit(0)
                    inputs.append(sample[field])
        else:
            inputs = [sample[self.input_field] for sample in X]

        logging.info("Number of samples: {}!".format(len(inputs)))

        np.random.shuffle(inputs)

        self.model = Word2Vec(inputs,
                         size=self.params["size"],
                         window=self.params["window"],
                         min_count=self.params["min_count"],
                         workers=self.params["workers"])
        self.model.save(str(self.emb_path))

    def apply_to_input_layer(self, input_layer, max_seq_len, apply_mask=True):

        if not self.weights_loaded:
            self.load_weights()

        emb = KerasEmb(input_dim=self.vocab_size,
                       output_dim=self.output_size,
                       embeddings_initializer=Constant(self.matrix),
                       input_length=max_seq_len)
        layer = emb(input_layer)

        if apply_mask:
            mask = Masking(mask_value=self.pad_idx,
                           input_shape=(max_seq_len, self.output_size))
            layer = mask(layer)

        return layer


