import logging
from datasets.parsers import FeatureParser, Word2VecParser, Code2VecParser, SafeParser
from datasets.source_dataset import SourceDataset


class SourcePipeline(object):

    handlers = {
        "word2vec": Word2VecParser,
        "code2vec": Code2VecParser,
        "safe": SafeParser
    }

    def __init__(self, args):
        self.args  = args
        self.steps = []

    def init_from_model_config(self):

        features     = self.args["features"]
        model_params = self.args["models"][self.args["model"]]
        inputs       = model_params["encoders"]["inputs"]
        self.steps   = []

        for input_kind in inputs:

            scenario_type   = input_kind["scenario"]
            scenario_params = features["scenarios"][scenario_type]
            encoder_type    = input_kind["encoder"]
            encoder_params  = self.args["encoders"][encoder_type]
            embedding_type  = scenario_params["type"]

            parser   = self.handlers[embedding_type](self.args,
                                                     scenario_params,
                                                     encoder_params)
            self.steps.append(parser)

        return self

    def prepare_data(self, samples):
        dataset = SourceDataset(self.args, samples)
        return dataset.prepare()

    def run(self, samples):

        X, X_meta = [], {}

        for step in self.steps:
            samples = step.filter_incompatible_samples(samples)
            if len(samples) == 0:
                raise Exception("{} features not available in the provided samples"
                                .format(step.features_kind))

        for step in self.steps:
            logging.info("Extracting {} features..".format(step.features_kind))
            result = step.extract_features(samples)

            if isinstance(result, tuple):
                x, x_meta = result
                meta_type = "{}_{}".format(step.features_kind, step.input_field)
                X_meta[meta_type] = x_meta
                X.append(x)
            elif isinstance(result, dict):
                for key in result:
                    X.append(result[key])
            else:
                X.append(result)

            logging.info("Extracting {} features.. Done!".format(step.features_kind))

        Y = self.get_outputs(samples)
        return (X, X_meta), Y

    def get_outputs(self, samples):

        Y = []
        for sample in samples:
            if "Y" not in sample:
                raise Exception("Output not available for sample {}".format(sample["index"]))
            Y.append(sample["Y"])
        return Y

    def __rshift__(self, extractor):

        if not isinstance(FeatureParser, extractor):
            raise NotImplementedError()
        self.steps.append(extractor)
        logging.info("Setup {} parser!".format(extractor.__name__))
        return self
