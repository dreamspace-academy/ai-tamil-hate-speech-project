# Training code

## Table of contents

1. [Description](#description)
2. [File structure](#file-structure)
3. [Training arguments](#training-arguments)
4. [Running](#running)

## Description

This pipeline is intended for performing training, validation of model.

## File structure

* **data/**
	* `dataset.py` : Pytorch Dataset object that loads training examples while training.
	* `helpers.py` : Several functions that help with data loading and preprocessing.
* **models/**
	* `transformer.py` : Pytorch model definition based on transformer architecture.
* **utils/**
	* `logger.py` : Custom python logger for printing training status and metrics in real time. Once training is finished logger output is saved in `model_artifacts/logfile.log`.
	* `utils.py` : Several functions that help setting up training environment like: setting mlflow credentials, setting seed for training reproducibility, model saving/loading, logging metrics to file and mlflow.
* **model_artifacts/**
    * `model_best.pt` : File with the model with best validation metric (model with highest validation recall is picked).
	* `args.pt` : File with all arguments used for training model. Can be loaded with `torch.load('args.pt')` function.
	* `logfile.log` : File with all the history of model training.
	* `test_labels_gold.txt` : File with gold standard labels in test dataset.
	* `test_labels_prob_pred.txt` : Predicted probability for each observation by the best model picked.
	* `test_pr_values.txt` : File with precision-recall curve values obtained form the best model predictions.
* `params.yaml` : File with all the arguments/parameters that can be modified for training.
* `train_eval.py` : Entry point for model training. Contains the main logic like: creation of model, optimizer, loss criterion and training loop.
* `requirements.txt` : File with necessary libraries to run training in google colab notebook.

## Training arguments

| Parameter name | Description | Values |
|--|--|--|
| seed | Random seed for experiment reproducibility | `Integer` |
| max_epochs | Maximum number of epochs | `Integer` |
| batch_size | Number of training examples used in each training step | `Integer` |
| num_workers | Number of threads in CPU to load data examples | `Integer` |
| lr | Learning rate | `Float` |
| lr_patience | Scheduler learning rate | `Float` |
| es_patience | Early stopping threshold | `Integer` |
| gradient_accumulation_steps | Number of steps to accumulate gradient before updating weights while training | `Integer` |
| tunning_metric | Metric to use for scheduler | `Accuracy`,`F1`,`Precision`,`Recall` |
| loss_type | Type of loss function to optimize | `ce` (cross-entropy),`weighted_ce` |
| use_fp16 | Allow FP16 training | `Boolean` |
| label_smoothing | Label smoothing value (only applicable for multiclass tasks) | `Float` |
| hidden_size | Embedding size of transformer model | `Integer` |
| max_length | Maximum of tokens to consider for input text | `Integer` |
| class_threshold | Threshold to transform predicted probability into 0 or 1 class | `Float` between 0 and 1 |
| model_path | Transformer model name. Path must follow HuggingFace model format | `google/muril-base-cased`,`xlm-roberta-base`,`bert-base-multilingual-uncased`, etc. |
| task_name | Name of task | `hate_speech_binary_classification` |
| dataset | Name of folder where data will be loaded from | `hate-speech-homophobia` |
| labels | Python list specifying order of labels | `['Non-Hate-Speech', 'Hate-Speech']` (for binary classification) |

## Running

Model training can be reproduced from this [Notebook](https://dagshub.com/Omdena/NYU/src/cross-validation/notebooks/Training_workflow.ipynb) which contains all the instructions.