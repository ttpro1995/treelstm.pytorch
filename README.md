
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
  -h, --help            show this help message and exit
  --data DATA           path to dataset
  --saved SAVED         path to save location
  --glove GLOVE         directory with GLOVE embeddings
  --tag_glove TAG_GLOVE
                        use tag embedding pretrained by glove
  --rel_glove REL_GLOVE
                        use rel embedding pretrained by glove
  --batchsize BATCHSIZE
                        batchsize for optimizer updates
  --epochs EPOCHS       number of total epochs to run
  --lr LR               initial learning rate
  --rho LR              rho for adadelta
  --emblr EMLR          initial word embedding learning rate (0 is turning
                        off)
  --tag_emblr EMLR      initial word embedding learning rate (0 is turning
                        off)
  --rel_emblr EMLR      initial word embedding learning rate (0 is turning
                        off)
  --wd WD               weight decay (default: 1e-4)
  --reg REG             l2 regularization (default: 1e-4)
  --optim OPTIM         optimizer (default: adagrad)
  --sgd_momentum SGD_MOMENTUM
                        sdg momentum param
  --sgd_dampening SGD_DAMPENING
                        sdg dampening param
  --sgd_nesterov SGD_NESTEROV
                        sdg nesterov param
  --lr_decay LR_DECAY   lr_decay (adagrad only)
  --scheduler SCHEDULER
                        use scheduler (default: 0, not use)
  --scheduler_factor SCHEDULER_FACTOR
                        factor
  --scheduler_patience SCHEDULER_PATIENCE
                        patience
  --scheduler_cooldown SCHEDULER_COOLDOWN
                        cooldown
  --scheduler_epsilon SCHEDULER_EPSILON
                        epsilon
  --grad_clip GRAD_CLIP
                        clipping gradient threshold (99999 for disable)
  --grad_noise GRAD_NOISE
                        gradient noise on off (default 0 off)
  --grad_noise_n GRAD_NOISE_N
                        grad noise n (in 0.01, 0.3, 1)
  --seed SEED           random seed (default: random)
  --fine_grain FINE_GRAIN
                        fine grained (default False)
  --input_dim INPUT_DIM
                        embedding dimension (default:30)
  --mem_dim MEM_DIM     mem_dim (default:150)
  --tag_dim TAG_DIM     tag embedding dimension (default:20)
  --at_hid_dim AT_HID_DIM
                        hidden dim of attention (0 for disable)
  --rel_dim REL_DIM     rel embedding dimension (default:0)
  --horizontal_dropout HORIZONTAL_DROPOUT
                        horizontal_dropout
  --leaf_dropout LEAF_DROPOUT
                        leaf dropout
  --vertical_dropout VERTICAL_DROPOUT
                        vertical dropout
  --word_dropout WORD_DROPOUT
                        word dropout
  --pos_tag_dropout POS_TAG_DROPOUT
                        pos tag dropout
  --output_module_dropout OUTPUT_MODULE_DROPOUT
                        sentiment module dropout (default: 0 disable)
  --name NAME           log name (default: default_log)
  --mode MODE           mode DEBUG, EXPERIMENT (default: DEBUG)
  --patient PATIENT     mode DEBUG, EXPERIMENT (default: DEBUG)
  --cuda
  --no-cuda

```

```
# best hyperparameters
python sentiment.py --name run4 --mode EXPERIMENT --optim adam --epochs 20 --scheduler 0 --lr 0.001 --wd 1e-6 --emblr 0.05 --tag_emblr 0.05 --tag_glove 1 --horizontal_dropout 0.3 --vertical_dropout 0.1 --word_dropout 0.4 --pos_tag_dropout 0.4 --output_module_dropout 1
```

### License
MIT