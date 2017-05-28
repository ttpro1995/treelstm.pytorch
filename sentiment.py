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
from model_comp_lstm import *
from model import *
from model_com_gru import TreeCompositionGRUSentiment
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
from config import parse_args, print_config
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer

from embedding_model import EmbeddingModel

import gc
import const
# MAIN BLOCK
def main():
    global args
    args = parse_args(type=1)
    print_config(args)
    const.show_setting()

    args.word_dim= args.input_dim
    if args.fine_grain:
        args.num_classes = 5 # 0 1 2 3 4
    else:
        args.num_classes = 3 # 0 1 2 (1 neutral)

    args.cuda = args.cuda and torch.cuda.is_available()
    # args.cuda = False
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    rel_token_files = [os.path.join(split, 'rels.txt') for split in [train_dir, dev_dir, test_dir]]
    tag_token_files = [os.path.join(split, 'tags.txt') for split in [train_dir, dev_dir, test_dir]]
    vocab_file = os.path.join(args.data,'vocab.txt')
    rel_vocab_file = os.path.join(args.data, 'relvocab.txt')
    tag_vocab_file = os.path.join(args.data, 'tagvocab.txt')
    build_vocab(token_files, vocab_file)
    build_vocab(rel_token_files, rel_vocab_file)
    build_vocab(tag_token_files, tag_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file)
    relvocab = Vocab(filename=rel_vocab_file)
    relvocab.add('head')
    rel_self_idx = [relvocab.labelToIdx['head']]
    tagvocab = Vocab(filename=tag_vocab_file)
    print('==> SST vocabulary size : %d ' % vocab.size())
    print('==> SST rel vocabulary size : %d ' % relvocab.size())
    print('==> SST tag vocabulary size : %d ' % tagvocab.size())


    # Load SST dataset splits

    is_preprocessing_data = False # let program turn off after preprocess data

    # train
    train_file = os.path.join(args.data,'sst_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SSTDataset(train_dir, vocab, tagvocab, relvocab, args.num_classes, args.fine_grain)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True

    # dev
    dev_file = os.path.join(args.data,'sst_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SSTDataset(dev_dir, vocab, tagvocab, relvocab, args.num_classes, args.fine_grain)
        torch.save(dev_dataset, dev_file)
        is_preprocessing_data = True

    # test
    test_file = os.path.join(args.data,'sst_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTDataset(test_dir, vocab, tagvocab, relvocab, args.num_classes, args.fine_grain)
        torch.save(test_dataset, test_file)
        is_preprocessing_data = True

    criterion = nn.CrossEntropyLoss()
    # initialize model, criterion/loss_function, optimizer
    model = TreeCompositionGRUSentiment(
                args.cuda, args.input_dim,
                args.tag_dim, args.rel_dim,
        args.mem_dim,3, criterion,
        combine_head=args.combine_head, rel_self=rel_self_idx,
        args = args
            )

    # embedding_model = nn.Embedding(vocab.size(), args.input_dim,
    #                             padding_idx=Constants.PAD)

    embedding_model = EmbeddingModel(args.cuda, vocab.size(), tagvocab.size(), relvocab.size(), args.word_dim, args.tag_dim, args.rel_dim)

    model.tree_module.set_embedding_model(embedding_model)
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam([
                {'params': model.parameters(), 'lr': args.lr, 'weight_decay' : args.wd},
               # {'params': embedding_model.parameters(), 'lr': args.emblr}
            ])
    elif args.optim=='adagrad':
        optimizer = optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr, 'weight_decay' : args.wd},
               # {'params': embedding_model.parameters(), 'lr': args.emblr}
            ])

    print ('optim '+ args.optim)
    print ('learning rate '+ str(args.lr))
    print ('embedding learning rate '+ str(args.emblr))
    print ('weight decay' + str(args.wd))

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
        emb = torch.zeros(vocab.size(),glove_emb.size(1))
        # zero out the embeddings for padding and other special words if they are absent in vocab
        # for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
        #     emb[idx].zero_()
        # torch.manual_seed(555)
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True # flag to quit
        print('done creating emb, quit')

    tag_emb_file = os.path.join(args.data, 'tag_embed.pth')
    if os.path.isfile(tag_emb_file):
        tag_emb = torch.load(tag_emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'tagglove'))
        print('==> TAG GLOVE vocabulary size: %d ' % glove_vocab.size())
        tag_emb = torch.zeros(tagvocab.size(), glove_emb.size(1))
        # zero out the embeddings for padding and other special words if they are absent in vocab
        # for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
        #     emb[idx].zero_()
        # torch.manual_seed(555)
        for word in tagvocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                tag_emb[tagvocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                tag_emb[tagvocab.getIndex(word)] = torch.Tensor(tag_emb[tagvocab.getIndex(word)].size()).normal_(-0.05, 0.05)
        torch.save(tag_emb, tag_emb_file)
        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')

    rel_emb_file = os.path.join(args.data, 'rel_embed.pth')
    if os.path.isfile(rel_emb_file):
        rel_emb = torch.load(rel_emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'relglove'))
        print('==> REL GLOVE vocabulary size: %d ' % glove_vocab.size())
        rel_emb = torch.zeros(relvocab.size(), glove_emb.size(1))
        # zero out the embeddings for padding and other special words if they are absent in vocab
        # for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
        #     emb[idx].zero_()
        # torch.manual_seed(555)
        for word in relvocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                rel_emb[relvocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                rel_emb[relvocab.getIndex(word)] = torch.Tensor(rel_emb[relvocab.getIndex(word)].size()).normal_(-0.05,
                                                                                                              0.05)
        torch.save(rel_emb, rel_emb_file)
        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')

    if is_preprocessing_data:
        print ('quit program due to memory leak during preprocess data, please rerun sentiment.py')
        quit()

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)
    # embedding_model.state_dict()['weight'].copy_(emb)
    # initialize word embeding

    # model.embedding_model.word_embedding.state_dict()['weight'].copy_(emb)
    embedding_model.word_embedding.state_dict()['weight'].copy_(emb)
    if args.tag_glove:
        assert args.tag_dim == 50
        embedding_model.tag_emb.state_dict()['weight'].copy_(tag_emb)
    if args.rel_glove:
        assert args.rel_dim == 50
        embedding_model.rel_emb.state_dict()['weight'].copy_(rel_emb)


    # create trainer object for training and testing
    trainer  = SentimentTrainer(args, model, embedding_model, criterion, optimizer)

    mode = args.mode
    print ('Run mode '+args.mode)
    if mode == 'DEBUG':
        stat_dev_loss = []
        stat_dev_acc = []
        stat_test_loss = []
        stat_test_acc = []
        for epoch in range(args.epochs):
            dev_loss = trainer.train(dev_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            test_loss, test_pred = trainer.test(test_dataset)


            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)

            stat_dev_loss.append(dev_loss)
            stat_dev_acc.append(dev_acc)
            stat_test_acc.append(test_acc)
            stat_test_loss.append(test_loss)

            utils.plot_loss(stat_dev_loss, stat_test_loss, args)
            utils.plot_accuracy(stat_dev_acc, stat_test_acc, args)

            print('==> Dev loss   : %f \t' % dev_loss, end="")
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
    elif mode == "EXPERIMENT":
        max_dev = 0
        max_dev_epoch = 0
        stat_train_loss = []
        stat_dev_loss = []
        stat_train_acc = []
        stat_dev_acc = []

        filename = args.name+'.pth'
        for epoch in range(args.epochs):
            train_loss             = trainer.train(train_dataset)
            train_loss, train_pred = trainer.test(train_dataset)
            dev_loss, dev_pred     = trainer.test(dev_dataset)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'train percentage ', train_acc)
            print('Epoch ',epoch, 'dev percentage ',dev_acc )
            stat_train_loss.append(train_loss)
            stat_dev_loss.append(dev_loss)
            stat_dev_acc.append(dev_acc)
            stat_train_acc.append(train_acc)

            utils.plot_loss(stat_train_loss, stat_dev_loss, args)
            utils.plot_accuracy(stat_train_acc, stat_dev_acc, args)

            if dev_acc > max_dev:
                max_dev = dev_acc
                max_dev_epoch = epoch
                torch.save(model, args.saved + str(epoch) + '_model_' + filename)
                torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
            gc.collect()
        print ('epoch ' + str(max_dev_epoch) +' dev score of ' + str(max_dev))
        print ('eva on test set ')
        model = torch.load(args.saved + str(max_dev_epoch)+'_model_'+filename)
        embedding_model = torch.load(args.saved + str(max_dev_epoch)+'_embedding_'+filename)
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
        test_loss, test_pred = trainer.test(test_dataset)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        print('Epoch with max dev:' + str(max_dev_epoch) + ' |test percentage '+ str(test_acc))
        print ('____________________'+str(args.name)+'___________________')
    else:
        for epoch in range(args.epochs):
            train_loss             = trainer.train(train_dataset)
            train_loss, train_pred = trainer.test(train_dataset)
            dev_loss, dev_pred     = trainer.test(dev_dataset)
            test_loss, test_pred   = trainer.test(test_dataset)

            # TODO: torch.Tensor(dev_dataset.labels) turn label into tensor # done
            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'train percentage ', train_acc)
            print('Epoch ',epoch, 'dev percentage ',dev_acc )
            print('Epoch ', epoch, 'test percentage ', test_acc)



if __name__ == "__main__":
    # log to console and file
    args = parse_args(type=1)
    utils.mkdir_p('plot')
    log_dir = os.path.join('plot', args.name)
    logger1 = log_util.create_logger(log_dir, print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    print ('_________________________________start___________________________________')
    main()