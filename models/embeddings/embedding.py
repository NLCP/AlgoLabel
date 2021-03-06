from gensim.models import Word2Vec
import numpy as np
from util import fcall
from pathlib import Path


class Embedding(object):

    def __init__(self, args, output_size, input_size = None):

        self.args        = args
        self.input_size  = input_size
        self.output_size = output_size

        self.emb_path = Path.cwd() / "data" / "embeddings"

        self.pad_vector  = np.zeros(output_size)
        self.matrix      = None
        self.vocab_size  = None
        self.unk_idx     = None
        self.pad_idx     = None

    def get_token_id(self, token):
        raise NotImplementedError()

    def load_weights(self, path):
        raise NotImplementedError()

    def pretrain(self, dataset):
        raise NotImplementedError()
