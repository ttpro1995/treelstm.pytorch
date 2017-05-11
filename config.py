import argparse

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
    else:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analyusis on Dependency Trees')
        parser.add_argument('--data', default='data/sst/',
                            help='path to dataset')
        parser.add_argument('--saved', default='saved_model/',
                            help='path to save location')
        parser.add_argument('--glove', default='data/glove/',
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
        parser.add_argument('--tag_dim', default=20, type=int,
                            help='tag embedding dimension (default:20)')
        parser.add_argument('--rel_dim', default=20, type=int,
                            help='rel embedding dimension (default:20)')
        parser.add_argument('--at_hid_dim', default=100, type=int,
                            help='hidden dim of attention (0 for disable)')

        parser.add_argument('--name', default='default_log',
                            help='log name (default: default_log)')

        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.set_defaults(cuda=True)

        args = parser.parse_args()
        return args

def print_config(args):
    print ('name ' + str(args.name))
    print ('batchsize ' + str(args.batchsize))
    print ('epochs ' + str(args.epochs))
    print ('lr ' + str(args.lr))
    print ('emblr ' + str(args.emblr))
    print ('tag_emblr ' + str(args.tag_emblr))
    print ('rel_emblr ' + str(args.rel_emblr))
    print ('wd ' + str(args.wd))
    print ('reg ' + str(args.reg))
    print ('optim ' + str(args.optim))
    print ('input_dim ' + str(args.input_dim))
    print ('mem_dim ' + str(args.mem_dim))

    print ('tag_dim ' + str(args.tag_dim))
    print ('rel_dim ' + str(args.rel_dim))
    print ('at_hid_dim ' + str(args.at_hid_dim))

    print ('tag_glove ' + str(args.tag_glove))
    print ('rel_glove ' + str(args.rel_glove))
