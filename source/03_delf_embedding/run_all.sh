#!/bin/bash

project=delf_embedding

python 01_train_embedding.py -n ${project}
python 02_valid_model.py -n ${project}
python 03_embed_index.py -n ${project}
python 04_make_submission.py -n ${project}
python 05_make_submission_1000.py -n ${project}
