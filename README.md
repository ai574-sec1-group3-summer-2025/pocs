
# Group 3's project for AI574, summer 2025.

Group Members:
* Balachander Srinivasan
* Mahfuzur Rahman
* Christopher Umbel

It comprises 3 tasks:
* Medical specialty classification
* Named-Entity Recognition of medical terms
* Transcript summarization

# Project Structure

* `/specialty_classifiction` - Classification of transcriptions according medical specialty
* `/summarization` - Summarization of medical transcriptions
* `/NER` - Named-Entity Recognition to extract relevant medical terms
* `/data` - Folder containing the main dataset as well as various processed outputs 

# Dataset

All tasks use the same dataset of 5000 medical transcriptions:

https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions/data

Each task directory contains a `requirements.txt` which can be installed with
`pip install -r ./requirements.txt`. It is highly recommended that each task
is run in its own Anaconda environment.

# Task Details

## Classification

* `classificiation.ipynb` - Main notebook for classification task
* `data_utils.py` - Utility class for cleaning and analyzing data

### Running
```
cd specialty_classifiction
jupyter lab
```

## NER

* `ner_model.ipynb` - Main, completely self-contained, notebook for NER task.

### Running
```
cd NER
jupyter lab
```


## Summarization

The following notebooks contain the project code
* `summarization.ipynb` - Main project notebook that summarizes the transcriptions and saves them to disk with three different pretrained models.
* `summarization-evaluation.ipynb` - Loads the machine-generated transcriptions
and performs various evaluation routines to determine the quality of each model's
output.

### Requirements
These models probably both require a CUDA video card in order to work. They could probably be changed to run on CPU, albeit significantly slower. 

### Login to Huggingface
`huggingface-cli login`
You'll need a Huggingface Account for anything using `transformers`. 
Create a READ token to login: https://huggingface.co/docs/hub/en/security-tokens

### Running
```
cd summarization
jupyter lab
```


