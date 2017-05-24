from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var

# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model_comp_lstm import SimilarityTreeLSTM
from embedding_model import EmbeddingModel
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SICKDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SimilarityTrainer
import utils
import gc
from meowlogtool import log_util
import sys

# MAIN BLOCK
def main():
    global args
    args = parse_args(type=10)
    args.input_dim = 300
    args.word_dim = args.input_dim
    args.hidden_dim = 50
    # args.input_dim, args.mem_dim = 300, 150
    # args.hidden_dim, args.num_classes = 50, 5
    if args.model_name == 'dependency':
        args.mem_dim = 150
    elif args.model_name == 'constituency':
        args.mem_dim = 142
    args.num_classes = 5

    args.cuda = args.cuda and torch.cuda.is_available()
    print(args)
    # torch.manual_seed(args.seed)
    # if args.cuda:
        # torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    token_files_a = [os.path.join(split,'a.toks') for split in [train_dir,dev_dir,test_dir]]
    token_files_b = [os.path.join(split,'b.toks') for split in [train_dir,dev_dir,test_dir]]
    tag_files_a = [os.path.join(split, 'a.tags') for split in [train_dir, dev_dir, test_dir]]
    tag_files_b = [os.path.join(split, 'b.tags') for split in [train_dir, dev_dir, test_dir]]
    rel_files_a = [os.path.join(split, 'a.rels') for split in [train_dir, dev_dir, test_dir]]
    rel_files_b = [os.path.join(split, 'b.rels') for split in [train_dir, dev_dir, test_dir]]

    token_files = token_files_a+token_files_b
    tag_token_files = tag_files_a + tag_files_b
    rel_token_files = rel_files_a + rel_files_b
    sick_vocab_file = os.path.join(args.data,'vocab-cased.txt')
    rel_vocab_file = os.path.join(args.data, 'relvocab.txt')
    tag_vocab_file = os.path.join(args.data, 'tagvocab.txt')
    build_vocab(rel_token_files, rel_vocab_file)
    build_vocab(tag_token_files, tag_vocab_file)
    # build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    relvocab = Vocab(filename=rel_vocab_file)
    tagvocab = Vocab(filename=tag_vocab_file)
    print('==> SICK rel vocabulary size : %d ' % relvocab.size())
    print('==> SICK tag vocabulary size : %d ' % tagvocab.size())
    print('==> SICK vocabulary size : %d ' % vocab.size())

    is_preprocessing_data = False

    # load SICK dataset splits
    train_file = os.path.join(args.data,'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, tagvocab, relvocab, args.num_classes)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True
    print('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data,'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, tagvocab, relvocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
        is_preprocessing_data = True
    print('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data,'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(test_dir, vocab, tagvocab, relvocab, args.num_classes)
        torch.save(test_dataset, test_file)
        is_preprocessing_data = True
    print('==> Size of test data    : %d ' % len(test_dataset))

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
                args.cuda, vocab.size(),
                args.word_dim, args.tag_dim,
                args.rel_dim, args.mem_dim, args.hidden_dim, 5
            )
    embedding_model = EmbeddingModel(args.cuda, vocab.size(), tagvocab.size(), relvocab.size(), args.word_dim, args.tag_dim, args.rel_dim)
    if args.cuda:
        embedding_model = embedding_model.cuda()

    criterion = nn.KLDivLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        optimizer   = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
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
    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer     = SimilarityTrainer(args, model, embedding_model, criterion, optimizer)

    # for epoch in range(args.epochs):
    #     train_loss             = trainer.train(train_dataset)
    #     train_loss, train_pred = trainer.test(train_dataset)
    #     dev_loss, dev_pred     = trainer.test(dev_dataset)
    #     test_loss, test_pred   = trainer.test(test_dataset)
    #
    #     print('==> Train loss   : %f \t' % train_loss, end="")
    #     print('Train Pearson    : %f \t' % metrics.pearson(train_pred,train_dataset.labels), end="")
    #     print('Train MSE        : %f \t' % metrics.mse(train_pred,train_dataset.labels), end="\n")
    #     print('==> Dev loss     : %f \t' % dev_loss, end="")
    #     print('Dev Pearson      : %f \t' % metrics.pearson(dev_pred,dev_dataset.labels), end="")
    #     print('Dev MSE          : %f \t' % metrics.mse(dev_pred,dev_dataset.labels), end="\n")
    #     print('==> Test loss    : %f \t' % test_loss, end="")
    #     print('Test Pearson     : %f \t' % metrics.pearson(test_pred,test_dataset.labels), end="")
    #     print('Test MSE         : %f \t' % metrics.mse(test_pred,test_dataset.labels), end="\n")

    mode = args.mode
    print('Run mode ' + args.mode)
    if mode == 'DEBUG':
        stat_dev_loss = []
        stat_dev_acc = []
        stat_test_loss = []
        stat_test_acc = []
        for epoch in range(args.epochs):
            dev_loss = trainer.train(dev_dataset, DEBUG=True)
            dev_loss, dev_pred = trainer.test(dev_dataset, DEBUG=True)
            test_loss, test_pred = trainer.test(test_dataset, DEBUG=True)

            dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels[0:5])
            dev_mse = metrics.mse(dev_pred, dev_dataset.labels[0:5])


            stat_dev_loss.append(dev_loss)
            stat_test_loss.append(test_loss)
            stat_dev_acc.append(dev_pearson)

            utils.plot_loss(stat_dev_loss, stat_test_loss, args)
            utils.plot_Pearson(stat_dev_acc)

            print('==> Dev loss   : %f \t' % dev_loss, end="")
            print('Epoch ', epoch, 'dev MSE ', dev_mse)
    elif mode == "EXPERIMENT":
        max_dev = -2
        max_dev_epoch = 0
        stat_train_loss = []
        stat_dev_loss = []
        stat_dev_acc = []

        filename = args.name + '.pth'
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            # train_loss, train_pred = trainer.test(train_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
            dev_acc = dev_pearson # let pearson be score
            dev_mse = metrics.mse(dev_pred, dev_dataset.labels)


            print('==> Train loss   : %f \t' % train_loss, end="")
            print ('Dev pearson ' + str(dev_pearson))
            print ('Dev MSE ' + str(dev_mse))
            stat_train_loss.append(train_loss)
            stat_dev_loss.append(dev_loss)
            stat_dev_acc.append(dev_pearson)


            utils.plot_loss(stat_train_loss, stat_dev_loss, args)
            utils.plot_Pearson(stat_dev_acc, args)

            if dev_acc > max_dev:
                max_dev = dev_acc
                max_dev_epoch = epoch
                torch.save(model, args.saved + str(epoch) + '_model_' + filename)
                print ('saved '+args.saved + str(epoch) + '_model_' + filename)
                torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
            gc.collect()
        print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))
        print('eva on test set ')
        model = torch.load(args.saved + str(max_dev_epoch) + '_model_' + filename)
        embedding_model = torch.load(args.saved + str(max_dev_epoch) + '_embedding_' + filename)
        trainer = SimilarityTrainer(args, model, embedding_model, criterion, optimizer)
        test_loss, test_pred = trainer.test(test_dataset)
        test_score = metrics.pearson(test_pred, test_dataset.labels)
        test_score_mse = metrics.mse(test_pred, test_dataset.labels)
        print('Epoch with max dev:' + str(max_dev_epoch) + ' |test score Pearson ' + str(test_score))
        print ('|test score MSE ' + str(test_score_mse))
        print('____________________' + str(args.name) + '___________________')

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

    main()