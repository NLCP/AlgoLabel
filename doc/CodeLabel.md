In order to simplify the various processing scripts, 
we use "config.json" to define the parameters of each operation.

##Steps:

If you don't want to preprocess new data you can skip straight to step 4.

An already preprocessed version of the dataset is available [here](https://drive.google.com/drive/u/0/folders/1rCu78ouGKhjLNyHML_R_gNjBpP3Utsvl).

Pre-trained models can be downloaded [here](https://drive.google.com/drive/u/0/folders/199zT4k-pHRiZS9gnzUdCTklj_c8spSC4).


### Step 1. Prepare

Preprocess raw source code data.

Prerequisites: 
- download and install [cppcheck](http://cppcheck.sourceforge.net/).

**Command**

	* "python ./main.py --prepare"

**Parameters** (config.json):

	* "raw_dataset" - the full path to the original crawled dataset.
	* "prepare"     - various preprocessing options

**Input**:

```
original_dataset = [
	{
		"index": """Problem Index"""
		"tags": ["""List of problem labels (e.g. dp, graphs, etc.)"""]
		"solutions": [
			{
				"code": """Original solution code"""
				"index": """Unique Solution Index"""
			}
		]
	}
]
```

* Remove non ASCII characters
* Remove external includes
* Remove comments and expand macros by running the compiler (g++) preprocessor
* Remove unused function calls (using cppcheck)
* Split source into tokens (using tokenizer)

**Output**:
```
prepared_dataset = [
	{
		"index": """Problem Index"""
		"tags": ["""List of problem labels (e.g. dp, graphs, etc.)"""]
		"solutions": [
			{
				"raw": """Original solution code"""
				"code": """Processed source code"""
				"tokens": """List of tokens"""
				"index": """Unique Solution Index"""
			}
		]
	}
]
```

### Step 2. Split
Splits the preprocessed dataset in train/dev/test/unlabeled.
Solutions belonging to the same problem appear in the same fold.

**Command**:

	"python ./main.py --split"

**Parameters** (config.json):

	* "split/percentage" - determines the amount of data used for "dev/test"
	* "split/labels"     
		- labels to account for when splitting the dataset
		- samples without these labels are grouped in a separate "unlabeled.json" file
	* "split/difficulty_based" 
		- account for the difficulty score (available for codeforces problems) during split
		- ensures an equal proportion of "easy/medium/hard" problems

**Output**:

Four files in the same folder as the original dataset. 
(e.g. "dataset_train.json", "dataset_test.json", "dataset_dev.json", "dataset_unlabeled.json")

Each files contains is of the form:

```
train = [
	{
		"raw": """Original solution code"""
		"code": """Processed source code"""
		"tokens": """List of tokens"""
		"index": """Unique Solution Index"""
	}
]
```

### Step 3. Pretrain

Computes several types of source code embeddings, depending on the value of the "pretrain" parameter
in config.json.

**Command**: 
	
	"python ./main.py --pretrain"

**Parameters** (config.json):

	"pretrain" - Valid values can be inspected in "features/scenarios".

Currently, we account for three types of features, that require special preprocessing: 

#### Step 3.1 Set "pretrain" == "word2vec_code" in config.json

Additional parameters:

	"features/types/word2vec"

What it does:
	* trains a word2vec model using the "train" and "unlabeled" split
	* saves the model in "data/embeddings/"

A pretrained word2vec model can be downloaded [here](https://drive.google.com/drive/u/0/folders/1y1Q2jpJEeYOZfqFBFe5iTxOdCehBpNRZ).

#### Step 3.2 Set "pretrain" == "code2vec"

A comprehensive description of the code2vec representation can be found in the [original paper](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf).

Additional parameters:

	"astminer" - path to astminer jar
	"features/types/code2vec"

What it does:
	
* Computes AST path contexts using [astminer](https://github.com/JetBrains-Research/astminer) for each file in one of the "train"/"dev"/"test" splits.
* Each path context is a tuple which consists of (start_token_id, path_index, end_token_id).
* Matches path contexts from the "dev"/"test" split to the indexes in "train".

#### Step 3.3. Set "pretrain" = "safe"

Additional Parameters: 
	
	"features/safe" 
	
What it does:

* Uses a [pretrained SAFE model](https://github.com/gadiluna/SAFE) to compute embeddings for each function in the available source files
* Note that the available model is built using a Tensorflow version <2.

### Step 4. Prepare Input

Compute Keras compatible input representations for each feature type according to the selected model.
Model configurations allow for many customization options.

**Command:**
	
	"python ./main.py --prepare_input"

**Parameter** (config.json): 

	"model" - the name of the model

**Available Parameters**: 

Any model name described in "models/" (config.json), e.g. "AlgoNetCode" 

**Output**:

Input and gold target samples for the chosen model, stored in "data/models/[model_name]/data/"
for each split.

### Step 5. Train

**Command:**
	
	"python ./main.py --train"

**Parameter** (config.json): 
	"model" - the name of the model
	"load_weight_from_epoch" - load weights from a custom epoch
	"train/num_epochs"
	"train/batch_size"

**Output**:

Trained Models are stored in "data/models/[model_name]"

### Step 6. Test

Uses the "test" split in "data/models/[model_name]/data/" to evaluate the model.

Model weights are loaded from "data/models/[model_name]" according to the
"load_weight_from_epoch" parameter (by default, )

**Command:**
	
	"python ./main.py --test"

**Parameter** (config.json): 

	"model" - the name of the model
	"load_weight_from_epoch" - load weights from a custom epoch
