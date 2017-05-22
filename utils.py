from __future__ import print_function

import os, math
import torch
from tree import Tree
from vocab import Vocab
import matplotlib.pyplot as plt

# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path+'.txt'))
    with open(path+'.txt','r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(map(float, contents[1:]))
            idx += 1
    with open(path+'.vocab','w') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors

# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename,'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile,'w') as f:
        for token in vocab:
            f.write(token+'\n')

# mapping from scalar to vector
def map_label_to_target(label,num_classes):
    target = torch.Tensor(1,num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - label
        target[0][ceil-1] = label - floor
    return target

def map_label_to_target_sentiment(label, num_classes = 0 ,fine_grain = False):
    # num_classes not use yet
    target = torch.LongTensor(1)
    target[0] = int(label) # nothing to do here as we preprocess data
    return target

def count_param(model):
    print('_param count_')
    params = list(model.parameters())
    sum_param = 0
    for p in params:
        sum_param+= p.numel()
        print (p.size())
    # emb_sum = params[0].numel()
    # sum_param-= emb_sum
    print ('sum', sum_param)
    print('____________')


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def plot_accuracy(train_accuracy, dev_accuracy, args, path='./plot/'):
    plt.subplots()
    x_axis = range(len(train_accuracy))
    plt.plot(x_axis, train_accuracy, 'r--',x_axis, dev_accuracy, 'b--')
    plt.title('Accuracy on Train vs on Dev : ' + args.name)
    out_dir = os.path.join(path, args.name)
    mkdir_p(out_dir)
    plt.savefig(os.path.join(out_dir, 'Accuracy' + '.png'))
    plt.close()

def plot_loss(train_loss, dev_loss, args, path='./plot/'):
    plt.subplots()
    x_axis = range(len(train_loss))
    plt.plot(x_axis, train_loss, 'r--',x_axis, dev_loss, 'b--')
    plt.title('Loss on Train vs on Dev : ' + args.name)
    out_dir = os.path.join(path, args.name)
    mkdir_p(out_dir)
    plt.savefig(os.path.join(out_dir, 'Loss' + '.png'))
    plt.close()

def plot_subtree_metrics(subtreemetric, epoch, args, plot_name = '', path='./plot/'):
    acc = subtreemetric.getAcc()
    plt.subplots()
    plt.title('Phrases acc: ' + args.name)
    phrases_length = []
    phrases_acc = []
    for key in acc.keys():
        phrases_length.append(key)
        phrases_acc.append(acc[key])

    plt.plot(phrases_length, phrases_acc)
    out_dir = os.path.join(path, args.name)
    mkdir_p(out_dir)
    plt.savefig(os.path.join(out_dir, 'Phrases' + '_'+str(epoch)+'_' +str(plot_name)+ '.png'))
    plt.close()


def plot_grad_stat_from_start(grads, grads_ratio, args, path='./plot/'):
    x_axis = range(len(grads))
    plt.figure(1)

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x_axis, grads, 'b--')
    axarr[0].set_title('Grad and Grad Ratio : ' + args.name)
    axarr[1].plot(x_axis, grads_ratio, 'r--')

    # plt.title('Grad and Grad Ratio : ' + args.name)
    # plt.subplot(211)
    # plt.plot(x_axis, grads, 'b--')
    #
    #
    # plt.subplot(212)
    # plt.plot(x_axis, grads_ratio, 'r--')
    out_dir = os.path.join(path, args.name)


    mkdir_p(out_dir)
    plt.savefig(os.path.join(out_dir, 'Grad' + '.png'))
    plt.close()

def plot_grad_stat_epoch(grads, grads_ratio, args, epoch, path='./plot/'):
    x_axis = range(len(grads))
    plt.figure(1)

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x_axis, grads, 'b--')
    axarr[0].set_title('Grad and Grad Ratio of epoch ' + str(epoch) + ' : ' + args.name)
    axarr[1].plot(x_axis, grads_ratio, 'r--')

    # plt.title('Grad and Grad Ratio of epoch ' + str(epoch) + ' : ' + args.name)
    # plt.subplot(211)
    # plt.plot(x_axis, grads, 'b--')
    #
    # plt.subplot(212)
    # plt.plot(x_axis, grads_ratio, 'r--')

    out_dir = os.path.join(path, args.name, 'Grad')


    mkdir_p(out_dir)
    plt.savefig(os.path.join(out_dir, str(epoch) + '.png'))
    plt.close()

def save_config(args, path='./plot/'):
    out_dir = os.path.join(path, args.name)
    mkdir_p(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'args.txt')):
        f = open(os.path.join(out_dir, 'args.txt'), 'w')
        f.write(str(args))
        f.close()
    else:
        f = open(os.path.join(out_dir, 'args.txt'), 'a')
        f.write('\n\n' + str(args))
        f.close()
        print ('args.txt file already exist, please, do manual check')