# IntentVizor

## Project Structure


```shell
./solvers # training solvers based on different settings.
./scripts # scripts for running the training
./models # models used in the project
./utils # utility code for the video summarization
./notebooks # notebooks for data preprocessing
./runners # runners for training the models
./factory # factory code for model and solvers (factory mode)
./loggers # training loggers
./exceptions # exceptions
./deployment # code for deploying the models.
./data # data loaders
./cache # code for the cache.
./evaluation # evaluation metrics
```

## Installation
```shell
pip install requirements.txt
```

## Running
The entrypoint of our project is the file train_qfvsum.py and train_sqfvsum.py. We also provide two bash scripts in scripts directory.
```shell
bash ./scripts/train_qfvsum.sh # train models on textual query dataset

bash ./scripts/train_vqfvsum.sh # train models on visual query dataset
```

## Guidance on Reading the Code
### Topic-Net and Intent-Net
In the early stage of developing the models, we use the term "topic" and "topic-net" to refer the currently used terms "intent" and "intent-net". 
As a result, the "topic" means "intent" in our paper. We do not change the term because the change of the name may make our trained model checkpoint fail.

### Model Directory
We divide the code into six sub-directories, i.e, baselines, feature_encoder, gcn, intent_net, score_net and visual_query. The model.py and shot_model.py are the entire models for textual and visual query datasets. 
The gcn directory stores the code for implementing GCN. Some of the code is collected from [G-TAD](https://github.com/frostinassiky/gtad).

### Visual Query Dataset Generation
We put the preprocessing code for visual query dataset in the notebooks directory.

### Checkpoint
We are sorry that we are not able to provide the checkpoint file due to the file limit on Microsoft CMT. However, we will make it publicly available after the review period.