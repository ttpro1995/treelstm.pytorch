

### Requirements
- [PyTorch](http://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 2.7

### Usage
First run the script `./fetch_and_preprocess.sh`, which downloads:
  - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!
  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)


```
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to dataset
  --saved SAVED         path to save location
  --glove GLOVE         directory with GLOVE embeddings
  --model_name MODEL_NAME
                        model name com_gru or childsumtree
  --test_idx TEST_IDX   dir to test idx np
  --embedding EMBEDDING
                        embedding type paragram or glove (default: glove)
  --paragram PARAGRAM   directory with paragram embeddings
  --tag_glove TAG_GLOVE
                        use tag embedding pretrained by glove
  --rel_glove REL_GLOVE
                        use rel embedding pretrained by glove
  --combine_head COMBINE_HEAD
                        node head (mid or end)
  --batchsize BATCHSIZE
                        batchsize for optimizer updates
  --epochs EPOCHS       number of total epochs to run
  --lr LR               initial learning rate
  --emblr EMLR          initial word embedding learning rate (0 is turning
                        off)
  --tag_emblr tag_EMLR  initial word embedding learning rate (0 is turning
                        off)
  --rel_emblr rel_EMLR  initial word embedding learning rate (0 is turning
                        off)
  --wd WD               weight decay (default: 1e-4)
  --embwd EMBWD         embedding weight decay (default: 1e-4)
  --tag_embwd TAG_EMBWD
                        tag embedding weight decay (default: 0)
  --rel_embwd REL_EMBWD
                        rel embedding weight decay (default: 0)
  --p_dropout_input P_DROPOUT_INPUT
                        p_dropout_input
  --p_dropout_memory P_DROPOUT_MEMORY
                        p_dropout_memory
  --reg REG             l2 regularization (default: 1e-4)
  --optim OPTIM         optimizer (default: adagrad)
  --seed SEED           random seed (default: 123)
  --fine_grain FINE_GRAIN
                        fine grained (default False)
  --input_dim INPUT_DIM
                        embedding dimension (default:30)
  --mem_dim MEM_DIM     mem_dim (default:150)
  --tag_dim TAG_DIM     tag embedding dimension (default:20)
  --rel_dim REL_DIM     rel embedding dimension (default:20)
  --at_hid_dim AT_HID_DIM
                        hidden dim of attention (0 for disable)
  --share_mlp SHARE_MLP
                        share com_mlp (0 for disable)
  --name NAME           log name (default: default_log)
  --mode MODE           mode DEBUG, EXPERIMENT (default: EXPERIMENT)
  --cuda
  --no-cuda
```

```
# best hyperparameters
python sentiment.py --name run_13_9 --optim adam --lr 0.001 --wd 1e-4 --emblr 0.05 --tag_emblr 0 --tag_dim 50 --tag_glove 1 --rel_glove 1 --rel_dim 50 --rel_emblr 0 --epochs 20 --combine_head mid --p_dropout_input 0.5 --p_dropout_memory 0.1
```

The preprocessing script also generates dependency parses of the SICK dataset using the
[Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).

To try the Dependency Tree-LSTM from the paper to predict similarity for pairs of sentences on the SICK dataset, run `python main.py` to train and test the model, and have a look at `config.py` for command-line arguments.

The first run takes a few minutes because the GLOVE embeddings for the words in the SICK vocabulary will need to be read and stored to a cache for future runs. In later runs, only the cache is read in during later runs.


### License
MIT