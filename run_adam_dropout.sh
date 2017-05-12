#!/bin/sh
source activate ml
python sentiment.py --name full_comlstm_mid_adam --optim adam --tag_dim 50 --tag_glove 1 --rel_glove 1 --rel_dim 50 --lr 0.0003 --emblr 0.00015 --tag_emblr 0.00015 --rel_emblr 0.00015 --wd 0.00003 --epochs 100