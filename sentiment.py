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
from model_c_com_lstm import TreeLSTMSentiment
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SSTConstituencyDataset, make_subtree, partition_dataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
import utils
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer
from lr_scheduler import ReduceLROnPlateau
from embedding_model import EmbeddingModel
from faster_gru import TreeGRUSentiment
import gc


# MAIN BLOCK
def main():
    global args
    args = parse_args(type=Constants.TYPE_constituency)


    args.word_dim = args.input_dim
    if args.fine_grain:
        args.num_classes = 5  # 0 1 2 3 4
    else:
        args.num_classes = 3  # 0 1 2 (1 neutral)

    args.cuda = args.cuda and torch.cuda.is_available()
    # args.cuda = False
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    token_files = [os.path.join(split, 'sents.toksc') for split in [train_dir, dev_dir, test_dir]]
    tag_token_files = [os.path.join(split, 'ctags.txt') for split in [train_dir, dev_dir, test_dir]]
    vocab_file = os.path.join(args.data, 'vocab.txt')
    tag_vocab_file = os.path.join(args.data, 'tagvocab.txt')
    build_vocab(token_files, vocab_file)
    build_vocab(tag_token_files, tag_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file)
    tagvocab = Vocab(filename=tag_vocab_file)
    print('==> SST vocabulary size : %d ' % vocab.size())
    print('==> SST tag vocabulary size : %d ' % tagvocab.size())

    # Load SST dataset splits

    is_preprocessing_data = False  # let program turn off after preprocess data

    # train
    train_file = os.path.join(args.data, 'sst_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
        # train_subtree_dataset = torch.load(train_subtree_file)
    else:
        train_dataset = SSTConstituencyDataset(train_dir, vocab, tagvocab, args.num_classes, args.fine_grain)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True
    gc.collect()

    # dev
    dev_file = os.path.join(args.data, 'sst_dev.pth')
    # if args.mode == "BABY":
    #     dev_subtree_file = os.path.join(args.data, 'sst_dev_sub.pth')
    if os.path.isfile(dev_file): # and os.path.isfile(dev_subtree_file):
        dev_dataset = torch.load(dev_file)
        # if args.mode == "BABY":
        #     dev_subtree_dataset = torch.load(dev_subtree_file)
    else:
        dev_dataset = SSTConstituencyDataset(dev_dir, vocab, tagvocab, args.num_classes, args.fine_grain)
        torch.save(dev_dataset, dev_file)
        # if args.mode == "BABY":
        #     dev_subtree_dataset = make_subtree(dev_dataset, tag_flag=True, rel_flag=False)
        #     torch.save(dev_subtree_dataset, dev_subtree_file)
        is_preprocessing_data = True

    # test
    test_file = os.path.join(args.data, 'sst_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTConstituencyDataset(test_dir, vocab, tagvocab, args.num_classes, args.fine_grain)
        torch.save(test_dataset, test_file)
        is_preprocessing_data = True

    criterion = nn.CrossEntropyLoss()
    # initialize model, criterion/loss_function, optimizer
    # model = TreeLSTMSentiment(
    #             args.cuda, args.input_dim,
    #             args.tag_dim,
    #     args.mem_dim, 3, criterion
    #         )

    model = TreeGRUSentiment(args.cuda, args.input_dim, args.tag_dim, args.mem_dim, 3, criterion)

    # embedding_model = nn.Embedding(vocab.size(), args.input_dim,
    #                             padding_idx=Constants.PAD)

    embedding_model = EmbeddingModel(args.cuda, vocab.size(), tagvocab.size(), 0, args.word_dim, args.tag_dim,
                                     args.rel_dim)

    if args.cuda:
        model.cuda(), criterion.cuda()

    scheduler = None
    if args.optim == 'adam':
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            # {'params': embedding_model.parameters(), 'lr': args.emblr}
        ])
    elif args.optim == 'adagrad':
        # optimizer = optim.Adagrad([
        #     {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
        #     # {'params': embedding_model.parameters(), 'lr': args.emblr}
        # ])
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr,
                                  lr_decay=args.lr_decay, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(params=model.parameters(),
                              lr=args.lr,
                              momentum=args.sgd_momentum,
                              dampening=args.sgd_dampening,
                              weight_decay=args.wd,
                              nesterov=args.sgd_nesterov)
    if args.optim == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(),
                                   lr=args.lr, rho=args.rho,
                                   weight_decay=args.wd)
    elif args.optim == 'adagrad_decade':
        optimizer = optim.Adagrad([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            # {'params': embedding_model.parameters(), 'lr': args.emblr}
        ])
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5,
                                      verbose=1, mode='min', cooldown=0, epsilon=5e-2)
    elif args.optim == 'adam_decade':
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            # {'params': embedding_model.parameters(), 'lr': args.emblr}
        ])
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5,
                                      verbose=1, mode='min', cooldown=0, epsilon=5e-2)
    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience,
                                      verbose=1, mode='min', cooldown=args.scheduler_cooldown,
                                      epsilon=args.scheduler_epsilon)

    print('optim ' + args.optim)
    print('learning rate ' + str(args.lr))
    print('embedding learning rate ' + str(args.emblr))
    print('weight decay' + str(args.wd))

    metrics = Metrics(args.num_classes)

    utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1))
        # zero out the embeddings for padding and other special words if they are absent in vocab
        # for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
        #     emb[idx].zero_()
        # torch.manual_seed(555)
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05, 0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')

    tag_emb_file = os.path.join(args.data, 'tag_embed.pth')
    if os.path.isfile(tag_emb_file):
        tag_emb = torch.load(tag_emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'ctagglove'))
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
                tag_emb[tagvocab.getIndex(word)] = torch.Tensor(tag_emb[tagvocab.getIndex(word)].size()).normal_(-0.05,
                                                                                                                 0.05)
        torch.save(tag_emb, tag_emb_file)
        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')

        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')

    if is_preprocessing_data:
        print('quit program due to memory leak during preprocess data, please rerun sentiment.py')
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
        assert False

    # create trainer object for training and testing
    trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer, scheduler=scheduler)
    # debug meow
    # loss, pred, dev_sub_metric = trainer.test(dev_dataset)
    # printlist = dev_sub_metric.print_list
    # utils.print_trees_file(args, vocab, tagvocab, dev_dataset, printlist, 'meow_debug')
    ##
    mode = args.mode
    stat_dev_acc = []
    stat_train_acc = []
    stat_dev_loss = []
    stat_train_loss = []
    utils.save_config(args)
    if mode == 'DEBUG':
        for epoch in range(args.epochs):
            dev_dataset[0][0].depth()
            # sorted_dev = sorted(dev_dataset, key=lambda x: x[0].depth())
            train_loss = trainer.train(dev_dataset)
            dev_loss, dev_pred, dev_sub_metric = trainer.test(dev_dataset)
            test_loss, test_pred, test_sub_metric = trainer.test(test_dataset)
            stat_train_loss.append(dev_loss)
            stat_dev_loss.append(test_loss)

            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            stat_train_acc.append(dev_acc)
            stat_dev_acc.append(test_acc)

            utils.plot_loss(stat_train_loss, stat_dev_loss, args)
            utils.plot_accuracy(stat_train_acc, stat_dev_acc, args)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('==> Dev loss   : %f \t' % dev_loss, end="")
            print ('Dev subtree metrics')
            dev_sub_metric.printAcc()
            print(' Test subtree metrics')
            test_sub_metric.printAcc()
            utils.plot_subtree_metrics(dev_sub_metric, epoch, args, 'dev')
            utils.plot_subtree_metrics(test_sub_metric, epoch, args, 'test')
            print('Epoch ', epoch + 1, 'dev percentage ', dev_acc, 'test percentage ', test_acc)
    elif mode == "PRINT_TREE":
        print_list = torch.load(os.path.join(args.saved, args.name + 'printlist.pth'))
        utils.print_trees_file(args, vocab, tagvocab, dev_dataset, print_list, name=args.name)
        #
        # for i in range(0, 10):
        #     ttree, tsent, ttag, trel, tlabel = dev_dataset[i]
        #     se = vocab.convertToLabels(tsent, Constants.UNK)
        #     sentences = str(i) +' '+ '.join(se)'
        #     print(sentences)
        #     utils.print_tree(vocab, tagvocab, tsent, ttree, 0)
        #
        #     print('_______________')
        quit()
    elif mode == "EXPERIMENT":
        max_dev = 0
        max_dev_epoch = 0
        filename = args.name + '.pth'
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            train_loss, train_pred, train_sub_metric = trainer.test(train_dataset)
            dev_loss, dev_pred, dev_sub_metric = trainer.test(dev_dataset)
            stat_train_loss.append(train_loss)
            stat_dev_loss.append(dev_loss)

            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            stat_train_acc.append(train_acc)
            stat_dev_acc.append(dev_acc)

            utils.plot_loss(stat_train_loss, stat_dev_loss, args)
            utils.plot_accuracy(stat_train_acc, stat_dev_acc, args)
            utils.plot_subtree_metrics(dev_sub_metric, epoch, args, 'dev')
            utils.plot_subtree_metrics(train_sub_metric, epoch, args, 'train')

            print('==> Train loss   : %f \t' % train_loss, end="")
            print('train percentage ' + str(train_acc))
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            if dev_acc > max_dev:
                utils.mkdir_p(args.saved)
                max_dev = dev_acc
                max_dev_epoch = epoch
                torch.save(model, args.saved + str(epoch) + '_model_' + filename)
                torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
                best_dev_embedding = args.name + str(epoch) + "embedding.pth"
                best_dev_model = args.name + str(epoch) + "model.pth"
                best_dev_optim_state = args.name + str(epoch) + "optim_state.pth"
                torch.save(embedding_model, os.path.join(args.saved, best_dev_embedding))
                torch.save(model, os.path.join(args.saved, best_dev_model))
                torch.save(trainer.optimizer.state_dict(), os.path.join(args.saved, best_dev_optim_state))
            gc.collect()
        print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))
        print('eva on test set ')
        best_dev_embedding = args.name + str(max_dev_epoch) + "embedding.pth"
        best_dev_model = args.name + str(max_dev_epoch) + "model.pth"
        model = torch.load(os.path.join(args.saved, best_dev_model))
        embedding_model = torch.load(os.path.join(args.saved, best_dev_embedding))
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
        test_loss, test_pred, test_sub_metric = trainer.test(test_dataset)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        utils.plot_subtree_metrics(test_sub_metric, epoch, args, 'test')
        print('Epoch with max dev:' + str(max_dev_epoch) + ' |test percentage ' + str(test_acc))
        print('____________________' + str(args.name) + '___________________')
        # print incorrect tree
        # print incorrect tree
        print_list = dev_sub_metric.print_list
        torch.save(print_list, os.path.join(args.saved, args.name + 'printlist.pth'))
    elif mode == "BABY":
        # part_index = dev_subtree_dataset.part_index
        # part_index.append(dev_subtree_dataset.size)
        depth_group = [5, 10, 15, 25]
        depth_group_2 = [1, 2, 4, 6, 10, 15, 25]
        depth_group = depth_group_2
        start_depth = 0
        best_dev_model = None
        best_dev_embedding = None
        best_dev_optim_state = None
        for i in  range(len(depth_group)): # 0 5 10 15 20 25
            if best_dev_model != None: # load best model (by dev) from last bucket
                model = torch.load(os.path.join(args.saved, best_dev_model))
                embedding_model =  torch.load(os.path.join(args.saved, best_dev_embedding))
                optimizer_state_dict = torch.load(os.path.join(args.saved, best_dev_optim_state))
                trainer.optimizer.load_state_dict(optimizer_state_dict)
                trainer.embedding_model = embedding_model
                trainer.model = model
                assert trainer.scheduler == None # well, no scheduler for sure


            depth = depth_group[i]
            print ('depth %d and below' %(depth))
            # start_idx = part_index[start_depth] # start from
            print('start %d end %d depth' % (start_depth, depth))
            # end_idx = part_index[depth + 1]
            # # from start depth to depth is [start_depth_index, depth+1 index)
            #
            # print('start index %d end index %d ' %(start_idx, end_idx))
            # part_dev_dataset = partition_dataset(dev_subtree_dataset, start_idx, end_idx)
            n_iter = 0
            max_dev_score = 0
            while n_iter < args.patient: # drop and move to next len when there is no improve
                print ('iter %d'%(n_iter))
                # sorted_dev = sorted(dev_dataset, key=lambda x: x[0].depth())
                train_loss = trainer.train(train_dataset, max_depth=depth)
                dev_loss, dev_pred, dev_sub_metric = trainer.test(dev_dataset, allow_neutral=True)

                dev_sub_metric.printAccDepth(start_depth, depth)
                _, dev_acc = dev_sub_metric.getAccDepth(start_depth, depth)
                full_dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
                print('dev percentage %f from depth %d to %d'% (dev_acc, start_depth, depth))
                print('Dev acc on full dev dataset %f' % (full_dev_acc))
                if (dev_acc - max_dev_score) > 0.005:
                    max_dev_score = dev_acc
                    n_iter = 0
                    print ('Update max dev score %f'%(max_dev_score))
                    best_dev_embedding = args.name+str(depth)+"embedding.pth"
                    best_dev_model = args.name + str(depth) + "model.pth"
                    best_dev_optim_state = args.name + str(depth) + "optim_state.pth"
                    torch.save(embedding_model, os.path.join(args.saved, best_dev_embedding))
                    torch.save(model, os.path.join(args.saved, best_dev_model))
                    torch.save(trainer.optimizer.state_dict(), os.path.join(args.saved, best_dev_optim_state))
                    print('saved '+best_dev_embedding + ' ' + best_dev_model + ' ' + best_dev_optim_state)
                else:
                    n_iter += 1
                print('==> Train loss   : %f \t' % train_loss, end="")
                print('==> Dev loss   : %f \t' % dev_loss, end="")
                utils.plot_subtree_metrics(dev_sub_metric, trainer.epoch, args, 'dev')

                # print incorrect tree
                print_list = dev_sub_metric.print_list
                utils.print_trees_file(args, vocab, tagvocab, dev_dataset, print_list, str(depth) + '_' + str(n_iter))

            start_depth = depth + 1  # for next iteration
        dev_loss, dev_pred, dev_sub_metric = trainer.test(dev_dataset)
        dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
        utils.plot_subtree_metrics(dev_sub_metric, trainer.epoch, args, 'dev_final')
        print("Baby final dev acc %f" % (dev_acc))

        test_loss, test_pred, test_sub_metric = trainer.test(test_dataset)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        utils.plot_subtree_metrics(test_sub_metric, trainer.epoch, args, 'test')
        print ("Baby final test acc %f" %(test_acc))
    else:
        print('DA FUCK IS THIS MODE : ' + str(args.mode))
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            train_loss, train_pred = trainer.test(train_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            test_loss, test_pred = trainer.test(test_dataset)

            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'train percentage ', train_acc)
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            print('Epoch ', epoch, 'test percentage ', test_acc)


if __name__ == "__main__":
    # log to console and file
    args = parse_args(type=Constants.TYPE_constituency)
    utils.mkdir_p('plot')
    log_dir = os.path.join('plot', args.name)
    utils.mkdir_p(log_dir) # maker folder
    log_dir = os.path.join(log_dir, args.name) # plot/experiment name/experiment_name.log
    logger1 = log_util.create_logger(log_dir, print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    print('_________________________________start___________________________________')
    main()
    log_link = log_util.up_gist(log_dir+'.log', args.name, __file__)
    print(log_link)
