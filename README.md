### 
Download http://nlp.stanford.edu/data/glove.840B.300d.zip and put glove.840B.300d.txt into data/glove 

Install some python package
pip install meowlogtool 
pip install tqdm 

Command to run
python sentiment.py --emblr 0 --rel_dim 0 --tag_dim 0 --optim adagrad --name basic --lr 0.05 --wd 1e-4 --at_hid_dim 0
