from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import utils

import sys
from meowlogtool import log_util


# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SSTDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer

# MAIN BLOCK
def main():
    global args
    args = parse_args(type=1)
    args.input_dim, args.mem_dim = 300, 168
    if args.fine_grain:
        args.num_classes = 5 # 0 1 2 3 4
    else:
        args.num_classes = 3 # 0 1 2 (1 neutral)
    args.cuda = args.cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    vocab_file = os.path.join(args.data,'vocab.txt')
    build_vocab(token_files, vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    print('==> SST vocabulary size : %d ' % vocab.size())

    # Load SST dataset splits

    is_preprocessing_data = False # let program turn off after preprocess data

    # train
    train_file = os.path.join(args.data,'sst_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True

    # dev
    dev_file = os.path.join(args.data,'sst_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain)
        torch.save(dev_dataset, dev_file)
        is_preprocessing_data = True

    # test
    test_file = os.path.join(args.data,'sst_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain)
        torch.save(test_dataset, test_file)
        is_preprocessing_data = True


    # initialize model, criterion/loss_function, optimizer
    model = TreeLSTMSentiment(
                args.cuda, vocab.size(),
                args.input_dim, args.mem_dim,
                args.num_classes
            )
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        optimizer   = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.840B.300d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(vocab.size(),glove_emb.size(1)).normal_(-0.05,0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
        is_preprocessing_data = True # flag to quit
        print('done creating emb, quit')

    if is_preprocessing_data:
        print ('quit program due to memory leak during preprocess data, please rerun sentiment.py')
        quit()

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer     = SentimentTrainer(args, model, criterion, optimizer)

    for epoch in range(args.epochs):
        train_loss             = trainer.train(train_dataset)
        # train_loss, train_pred = trainer.test(dev_dataset)
        dev_loss, dev_pred     = trainer.test(dev_dataset)
        test_loss, test_pred   = trainer.test(test_dataset)
        # TODO: torch.Tensor(dev_dataset.labels) turn label into tensor # done
        dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ',epoch, 'dev percentage ',dev_acc )
        print('Epoch ', epoch, 'test percentage ', test_acc)



if __name__ == "__main__":
    # log to console and file
    logger1 = log_util.create_logger("temp_file", print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    print ('_________________________________start___________________________________')
    main()