import argparse
import random

def parse_args(type=0):
    if type == 0:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
        parser.add_argument('--data', default='data/sick/',
                            help='path to dataset')
        parser.add_argument('--glove', default='data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=15, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.01, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--optim', default='adam',
                            help='optimizer (default: adam)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.set_defaults(cuda=True)

        args = parser.parse_args()
        return args
    elif type == 1:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analyusis on Dependency Trees')
        parser.add_argument('--data', default='data/sst/',
                            help='path to dataset')
        parser.add_argument('--saved', default='saved_model/',
                            help='path to save location')
        parser.add_argument('--glove', default='../treelstm.pytorch/data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--tag_glove', default=False, type=bool,
                            help='use tag embedding pretrained by glove')
        parser.add_argument('--rel_glove', default=False, type=bool,
                            help='use rel embedding pretrained by glove')

        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=10, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0, type=float,
                            metavar='EMLR', help='initial word embedding learning rate (0 is turning off)')
        parser.add_argument('--tag_emblr', default=0, type=float,
                            metavar='EMLR', help='initial word embedding learning rate (0 is turning off)')
        parser.add_argument('--rel_emblr', default=0, type=float,
                            metavar='EMLR', help='initial word embedding learning rate (0 is turning off)')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=1e-4, type=float,
                            help='l2 regularization (default: 1e-4)')
        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        parser.add_argument('--fine_grain', default=False, type=bool,
                            help='fine grained (default False)')

        parser.add_argument('--input_dim', default=300, type=int,
                            help=' embedding dimension (default:30)')
        parser.add_argument('--mem_dim', default=150, type=int,
                            help='mem_dim (default:150)')
        parser.add_argument('--tag_dim', default=50, type=int,
                            help='tag embedding dimension (default:50)')
        parser.add_argument('--rel_dim', default=0, type=int,
                            help='rel embedding dimension (default:0)')
        parser.add_argument('--at_hid_dim', default=100, type=int,
                            help='hidden dim of attention (0 for disable)')

        parser.add_argument('--name', default='default_log',
                            help='log name (default: default_log)')

    elif type == 2: # Constituency Trees use this one
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analyusis on Constituency Trees')
        parser.add_argument('--data', default='data/sst/',
                            help='path to dataset')
        parser.add_argument('--saved', default='saved_model/',
                            help='path to save location')
        parser.add_argument('--glove', default='../treelstm.pytorch/data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--tag_glove', default=0, type=int,
                            help='use tag embedding pretrained by glove')
        parser.add_argument('--rel_glove', default=False, type=bool,
                            help='use rel embedding pretrained by glove')

        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=50, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--rho', default=0.95, type=float,
                            metavar='LR', help='rho for adadelta')
        parser.add_argument('--emblr', default=0, type=float,
                            metavar='EMLR', help='initial word embedding learning rate (0 is turning off)')
        parser.add_argument('--tag_emblr', default=0, type=float,
                            metavar='EMLR', help='initial word embedding learning rate (0 is turning off)')
        parser.add_argument('--rel_emblr', default=0, type=float,
                            metavar='EMLR', help='initial word embedding learning rate (0 is turning off)')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=1e-4, type=float,
                            help='l2 regularization (default: 1e-4)')

        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')

        parser.add_argument('--sgd_momentum', default=0, type=float,
                            help='sdg momentum param')
        parser.add_argument('--sgd_dampening', default=0, type=float,
                            help='sdg dampening param')
        parser.add_argument('--sgd_nesterov', default=0, type=int,
                            help='sdg nesterov param')

        parser.add_argument('--lr_decay', default=0, type=float,
                            help='lr_decay (adagrad only)')

        parser.add_argument('--scheduler', default=0, type=int,
                            help='use scheduler (default: 0, not use)')
        parser.add_argument('--scheduler_factor', default=0.1, type=float,
                            help='factor')
        parser.add_argument('--scheduler_patience', default=5, type=int,
                            help='patience')
        parser.add_argument('--scheduler_cooldown', default=0, type=float,
                            help='cooldown')
        parser.add_argument('--scheduler_epsilon', default=1e-4, type=float,
                            help='epsilon')

        parser.add_argument('--grad_clip', default=1000, type=float,
                            help='clipping gradient threshold (99999 for disable)')

        parser.add_argument('--grad_noise', default=1, type=int,
                            help='gradient noise on off (default 0 off)')
        parser.add_argument('--grad_noise_n', default=0.01, type=float,
                            help='grad noise n (in 0.01, 0.3, 1)')

        parser.add_argument('--seed', default=int(random.random()*1e+9), type=int,
                            help='random seed (default: random)')
        parser.add_argument('--fine_grain', default=False, type=bool,
                            help='fine grained (default False)')

        parser.add_argument('--input_dim', default=300, type=int,
                            help=' embedding dimension (default:30)')
        parser.add_argument('--mem_dim', default=150, type=int,
                            help='mem_dim (default:150)')
        parser.add_argument('--tag_dim', default=50, type=int,
                            help='tag embedding dimension (default:20)')
        parser.add_argument('--at_hid_dim', default=100, type=int,
                            help='hidden dim of attention (0 for disable)')

        parser.add_argument('--rel_dim', default=0, type=int,
                            help='rel embedding dimension (default:0)')

        parser.add_argument('--horizontal_dropout', default=0, type=float,
                            help='horizontal_dropout')
        parser.add_argument('--leaf_dropout', default=0, type=float,
                            help='leaf dropout')
        parser.add_argument('--vertical_dropout', default=0, type=float,
                            help='vertical dropout')
        parser.add_argument('--word_dropout', default=0, type=float,
                            help='word dropout')
        parser.add_argument('--pos_tag_dropout', default=0, type=float,
                            help='pos tag dropout')


        parser.add_argument('--output_module_dropout', default=0, type=int,
                            help='sentiment module dropout (default: 0 disable)')

        parser.add_argument('--name', default='default_log',
                            help='log name (default: default_log)')
        parser.add_argument('--mode', default='DEBUG',
                            help='mode DEBUG, EXPERIMENT (default: DEBUG)')
        parser.add_argument('--patient', default=1, type = int,
                            help='mode DEBUG, EXPERIMENT (default: DEBUG)')

    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args


