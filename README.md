# AlgoLabel

## Changelog

## Requirements

### Text Preprocessing

* [NLTK](https://www.nltk.org/)

### Code Preprocessing

* [Optional] [clang-format](https://clang.llvm.org/docs/ClangFormat.html)  
* [cppcheck](http://cppcheck.sourceforge.net/)
* [tokenizer](https://github.com/dspinellis/tokenizer)

### Neural baselines and evaluation

* [numpy](https://numpy.org/doc/stable/index.html)
* [gensim](https://radimrehurek.com/gensim/)
  - Word2Vec embeddings
* [Scikit-multilearn](http://scikit.ml/)
  - Multi-label data stratification 
* [TensorFlow 2](https://www.tensorflow.org/install)
* [Keras](https://keras.io/)
  - Neural baselines
  
### Other
* [tqdm](https://github.com/tqdm/tqdm)

## Citation


## Data Content and Format

Sample:

```yaml
    {
        index            : "(string) Unique sample index",
        tags             : "(list of strings) Complete list of associated labels",
        source           : "(string) Sample source",
        url              : "(string) Problem link",
        title            : "(string) Problem title",
        statement        : "(string) Description of problem statement (formulas are replaced with placeholders)",
        input            : "(string) Description of problem input (formulas are replaced with placeholders)",
        output           : "(string) Description of problem output (formulas are replaced with placeholders)",
        variables        : "(list of strings) List of detected variable names",
        contest_name     : "(string) Contest where the problem originally appeared",
        contest_link     : "(string) Link to the contest page",
        time_limit       : "(string) Time limit",
        memory_limit     : "(string) Memory limit",
        sentences        : [
          statement : "(list of strings) Statement split into sentences",
          input     : "(list of strings) Input split into sentences"
          output    : "(list of strings) Output split into sentences"          
        ]
        formulas         : "(dictionary) A map between formulas appearing in text and associated placeholders"
        formulas_idx     : "(dictionary) A map between placeholders and the associated formulas",
        difficulty       : "[optional](string) Difficulty value,
        difficulty_class : "[optional](string) Tag describing difficulty (Easy-Medium-Hard),
        hint             : "[optional] Problem hint",
        stats            : "[optional] Submission stats (number of submissions, rate of successful submissions, etc.)
        solutions        : [
          {
            "code"   : "[optional] Raw source code (C++)",
            "url"    : "[optional] Link to submission",
            "index"  : "[optional] Unique index",
            "tokens" : "[optional] Code Tokens",
            "symbs"  : "[optional] Annonymized Code Tokens",
            "safe"   : "[optional] (list of floats) SAFE code embeddings for each function"
          }
        ]
    }
```

## Evaluation

## Baselines

### AlgoLabelNet
### AlgoCode

## Dataset

The dataset is available at https://drive.google.com/drive/u/0/folders/1C_1AmEIfp0ZPqPUipOn2NeXTVIj7g4NF .

## FAQ

## License

This project is licensed under the Creative Commons Attribution 4.0 license.
( https://creativecommons.org/licenses/by/4.0/ )
