
This is group 3's project for AI574, summer 2025.

It comprises 3 tasks, which are implemented in the following directories. 

# Project Structure
* `/specialty_classifiction` - Classification of transcriptions according medical specialty
* `/summarization` - Summarization of medical transcriptions
* `/NER` - Named-Entity Recognition to extract relevant medical terms

# Dataset
All tasks use the same dataset of 5000 medical transcriptions:

https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions/data

Each task directory contains a `requirements.txt` which can be installed with
`pip install -r ./requirements.txt`. It is highly recommended that each task
is run in its own Anaconda environment.

# Task Details

## Classification

## Summarization

The following notebooks contain the project code
* `summarization.ipynb` - Main project notebook that summarizes the transcriptions and saves them to disk with three different pretrained models.
* `summarization-evaluation.ipynb` - Loads the machine-generated transcriptions
and performs various evaluation routines to determine the quality of each model's
output.

### Running
`cd summarization`
`jupyter lab`

### Requirements
These models probably both require a CUDA video card in order to work. They could probably be changed to run on CPU, albeit significantly slower. 

### Login to Huggingface
`huggingface-cli login`
You'll need a Huggingface Account for anything using `transformers`. 
Create a READ token to login: https://huggingface.co/docs/hub/en/security-tokens



