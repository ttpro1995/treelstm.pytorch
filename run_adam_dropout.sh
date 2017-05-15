#!/bin/sh
source activate ml
python sentiment.py --name full_comlstm_mid_adagad_norel_19_15_14_5 --optim adagrad --tag_dim 50 --tag_glove 1 --rel_dim 50 --rel_glove 1 --lr 0.05 --emblr 0.05 --tag_emblr 0.05 --rel_emblr 0 --wd 1e-4 --epochs 50

