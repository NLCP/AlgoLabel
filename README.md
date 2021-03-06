# AlgoLabel

## Changelog

05.03.2021
  - Significant code refactorization
  - Added a preprocessing step for the code-labeling task (macros are now correctly expanded)

## Requirements

### Text Preprocessing

* [NLTK](https://www.nltk.org/)

### Code Preprocessing

* The GNU Compiler Collection (g++)
* [Optional] [clang-format](https://clang.llvm.org/docs/ClangFormat.html)  
* [cppcheck](http://cppcheck.sourceforge.net/)
* [tokenizer](https://github.com/dspinellis/tokenizer)
* [astminer](https://github.com/JetBrains-Research/astminer)

### Code embeddings

* [SAFE](https://github.com/gadiluna/SAFE)
* [Code2Vec](https://github.com/tech-srl/code2vec) (re-implemented)

### Neural Baselines and Evaluation

* [numpy](https://numpy.org/doc/stable/index.html)
* [gensim](https://radimrehurek.com/gensim/)
  - Word2Vec embeddings
* [Scikit-multilearn](http://scikit.ml/)
  - Multi-label data stratification
* [TensorFlow 2](https://www.tensorflow.org/install)
  - Note: Pre-trained SAFE model was built using Tensorflow 1
* [Keras](https://keras.io/)
  - Neural baselines
  
### Other
* [tqdm](https://github.com/tqdm/tqdm)

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
      input     : "(list of strings) Input split into sentences",
      output    : "(list of strings) Output split into sentences"
    ]
    formulas         : "(dictionary) A map between formulas appearing in text and associated placeholders"
    formulas_idx     : "(dictionary) A map between placeholders and the associated formulas",
    difficulty       : "[optional](string) Difficulty value",
    difficulty_class : "[optional](string) Tag describing difficulty (Easy-Medium-Hard)",
    hint             : "[optional] Problem hint",
    stats            : "[optional] Submission stats (number of submissions, rate of successful submissions, etc.)",
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

### Text Label Task

A detailed description of the necessary steps can be inspected [here](doc/TextLabel.md).

| Model | F1 (micro-avg.) |
| ---------- | ---------------- |
| AlgoLabelNet | 0.62 |

### Code Label Task

A detailed description of the necessary steps can be inspected [here](doc/CodeLabel.md). 
Pre-trained models can be downloaded [here](https://drive.google.com/drive/u/0/folders/199zT4k-pHRiZS9gnzUdCTklj_c8spSC4): 

#### Code Label Results

Targets labels: math, implementation, graphs, dp & greedy. 

| Model                               | F1 (micro-avg.) |
| ---------- | ---------------- |
| AlgoCode (AST(200)) | 0.47 |
| AlgoCode (SAFE) | 0.50 |
| AlgoCode (Tokens) | 0.55 |
| AlgoCode (AST(200) + SAFE + Tokens) | 0.56 | 

## Dataset

The complete dataset is available [here](https://drive.google.com/drive/u/0/folders/1C_1AmEIfp0ZPqPUipOn2NeXTVIj7g4NF).

## FAQ

## Citation

```
@Article{math8111995,
AUTHOR = {Iacob, Radu Cristian Alexandru and Monea, Vlad Cristian and Rădulescu, Dan and Ceapă, Andrei-Florin and Rebedea, Traian and Trăușan-Matu, Ștefan},
TITLE = {AlgoLabel: A Large Dataset for Multi-Label Classification of Algorithmic Challenges},
JOURNAL = {Mathematics},
VOLUME = {8},
YEAR = {2020},
NUMBER = {11},
ARTICLE-NUMBER = {1995},
URL = {https://www.mdpi.com/2227-7390/8/11/1995},
ISSN = {2227-7390},
DOI = {10.3390/math8111995}
}
```

## License

This project is licensed under the 
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
