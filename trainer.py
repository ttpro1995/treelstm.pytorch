from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target, map_label_to_target_sentiment
import torch.nn.functional as F
import gc
import utils

class SentimentTrainer(object):
    """
    For Sentiment module
    """
    def __init__(self, args, model, embedding_model, criterion, optimizer):
        super(SentimentTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.embedding_model = embedding_model
        self.plot_tree_grad = []
        self.plot_tree_grad_param = []
        if self.args.model_name == 'com_gru':
            self.rel_self = model.rel_self  # rel self index



    # helper function for training
    def train(self, dataset, plot = True):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        print ('Start training epoch ' + str(self.epoch))

        epoch_plot_tree_grad = []
        epoch_plot_tree_grad_param = []

        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        total_sample = len(dataset)
        done_sample = 0
        for idx in tqdm(xrange(len(dataset))):
            tree, sent, tag, rel, label = dataset[indices[idx]]
            input = Var(sent)
            tag_input = Var(tag)
            rel_input = Var(rel)
            if self.args.model_name == 'com_gru':
                rel_self = Var(torch.Tensor(self.model.rel_self).long())
            if self.args.cuda:
                input = input.cuda()
                tag_input = tag_input.cuda()
                rel_input = rel_input.cuda()
                if self.args.model_name == 'com_gru':
                    rel_self = rel_self.cuda()
            sent_emb, tag_emb, rel_emb = self.embedding_model(input, tag_input, rel_input)
            if self.args.model_name == 'com_gru':
                rel_self = self.embedding_model.forward(None, None, rel_self)
                rel_self = rel_self[2]
            else:
                rel_self = None
            output, err = self.model.forward(tree, sent_emb, tag_emb, rel_emb, training = True, rel_self = rel_self)
            #params = self.model.get_tree_parameters()
            #params_norm = params.norm()
            err = err/self.args.batchsize #+ 0.5*self.args.reg*params_norm*params_norm # custom bias
            loss += err.data[0] #
            err.backward()
            k += 1
            #params = None
            #params_norm = None
            done_sample += 1
            # if done_sample % 1000 == 0:
            #     print ('epoch '+ str(self.epoch) + ' '+ str(done_sample) + '/'+str(total_sample))

            if k==self.args.batchsize:
                tree_model_param_norm = self.model.tree_module.getParameters().norm().data[0]
                tree_model_grad_norm = self.model.tree_module.getGrad().norm().data[0]
                tree_model_grad_param = tree_model_grad_norm / tree_model_param_norm

                self.plot_tree_grad.append(tree_model_grad_norm)
                epoch_plot_tree_grad.append(tree_model_grad_norm)
                self.plot_tree_grad_param.append(tree_model_grad_param)
                epoch_plot_tree_grad_param.append(tree_model_grad_param)

                if self.args.tag_emblr > 0 and self.args.tag_dim > 0:
                    for f in self.embedding_model.tag_emb.parameters(): # train tag embedding
                        f.data.sub_(f.grad.data * self.args.tag_emblr + self.args.tag_emblr*self.args.tag_embwd*f.data)

                if self.args.rel_emblr > 0 and self.args.rel_dim > 0:
                    for f in self.embedding_model.rel_emb.parameters():
                        f.data.sub_(f.grad.data * self.args.rel_emblr + self.args.rel_emblr*self.args.rel_embwd*f.data)

                if self.args.emblr > 0:
                    for f in self.embedding_model.word_embedding.parameters():
                        f.data.sub_(f.grad.data * self.args.emblr + self.args.emblr*self.args.embwd*f.data)

                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0

        if plot:
            utils.plot_grad_stat_epoch(epoch_plot_tree_grad, epoch_plot_tree_grad_param,
                                       self.args,self.epoch)
            utils.plot_grad_stat_from_start(self.plot_tree_grad, self.plot_tree_grad_param,
                                        self.args)

        self.epoch += 1
        gc.collect()
        print ('Done training epoch ' + str(self.epoch))
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        predictions = predictions
        indices = torch.range(1,dataset.num_classes)
        for idx in tqdm(xrange(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            tree, sent, tag, rel, label = dataset[idx]
            input = Var(sent, volatile=True)
            tag_input = Var(tag, volatile=True)
            rel_input = Var(rel, volatile=True)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.model_name == 'com_gru':
                rel_self = Var(torch.Tensor(self.model.rel_self).long())
            if self.args.cuda:
                input = input.cuda()
                tag_input = tag_input.cuda()
                rel_input = rel_input.cuda()
                target = target.cuda()
                if self.args.model_name == 'com_gru':
                    rel_self = rel_self.cuda()
            sent_emb, tag_emb, rel_emb = self.embedding_model(input, tag_input, rel_input)
            if self.args.model_name == 'com_gru':
                rel_self = self.embedding_model.forward(None, None, rel_self)
                rel_self = rel_self[2]
            else:
                rel_self = 0
            output, _ = self.model(tree, sent_emb, tag_emb, rel_emb, rel_self = rel_self) # size(1,5)
            err = self.criterion(output, target)
            loss += err.data[0]
            output[:,1] = -9999 # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            predictions[idx] = pred.data.cpu()[0][0]
            # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions

class SimilarityTrainer(object):
    def __init__(self, args, model, embedding_model, criterion, optimizer):
        super(SimilarityTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.embedding_model = embedding_model
        self.plot_tree_grad = []
        self.plot_tree_grad_param = []

    # helper function for training
    def train(self, dataset, plot = True, DEBUG = False):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        n_dataset = len(dataset)
        if DEBUG:
            n_dataset = 5
        indices = torch.randperm(n_dataset)
        epoch_plot_tree_grad = []
        epoch_plot_tree_grad_param = []

        for idx in tqdm(xrange(n_dataset),desc='Training epoch '+str(self.epoch+1)+''):
            ltree,lsent,rtree,rsent,ltag,rtag,lrel,rrel,label = dataset[indices[idx]]
            linput, rinput, ltag, rtag, lrel, rrel = Var(lsent), Var(rsent), Var(ltag), Var(rtag), Var(lrel), Var(rrel)
            target = Var(map_label_to_target(label,dataset.num_classes))
            if self.args.cuda:
                linput, rinput, ltag, rtag, lrel, rrel = linput.cuda(), rinput.cuda(), ltag.cuda(), rtag.cuda(), lrel.cuda(), rrel.cuda()
                target = target.cuda()
            lemb, ltagemb, lrelemb = self.embedding_model(linput, ltag, lrel)
            remb, rtagemb, rrelemb = self.embedding_model(rinput, rtag, rrel)
            output = self.model.forward(ltree, lemb, ltagemb, lrelemb, rtree, remb, rtagemb, rrelemb)
            err = self.criterion(output, target)
            loss += err.data[0]
            err.backward()
            k += 1
            if k==self.args.batchsize:
                tree_model_param_norm = self.model.tree_module.getParameters().norm().data[0]
                tree_model_grad_norm = self.model.tree_module.getGrad().norm().data[0]
                tree_model_grad_param = tree_model_grad_norm / tree_model_param_norm

                self.plot_tree_grad.append(tree_model_grad_norm)
                epoch_plot_tree_grad.append(tree_model_grad_norm)
                self.plot_tree_grad_param.append(tree_model_grad_param)
                epoch_plot_tree_grad_param.append(tree_model_grad_param)

                if self.args.tag_emblr > 0 and self.args.tag_dim > 0:
                    for f in self.embedding_model.tag_emb.parameters(): # train tag embedding
                        f.data.sub_(f.grad.data * self.args.tag_emblr)

                if self.args.rel_emblr > 0 and self.args.rel_dim > 0:
                    for f in self.embedding_model.rel_emb.parameters():
                        f.data.sub_(f.grad.data * self.args.rel_emblr)

                if self.args.emblr > 0:
                    for f in self.embedding_model.word_embedding.parameters():
                        f.data.sub_(f.grad.data * self.args.emblr)

                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
            if plot:
                utils.plot_grad_stat_epoch(epoch_plot_tree_grad, epoch_plot_tree_grad_param,
                                           self.args, self.epoch)
                utils.plot_grad_stat_from_start(self.plot_tree_grad, self.plot_tree_grad_param,
                                                self.args)
        self.epoch += 1
        return loss/n_dataset

    # helper function for testing
    def test(self, dataset, DEBUG = False):
        self.model.eval()
        loss = 0
        n_dataset = len(dataset)
        if DEBUG:
            n_dataset = 5
        predictions = torch.zeros(n_dataset)
        indices = torch.range(1,dataset.num_classes)
        for idx in tqdm(xrange(n_dataset),desc='Testing epoch  '+str(self.epoch)+''):
            # ltree,lsent,rtree,rsent,label = dataset[idx]
            # linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            # target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
            ltree,lsent,rtree,rsent,ltag,rtag,lrel,rrel,label = dataset[idx]
            linput, rinput, ltag, rtag, lrel, rrel = Var(lsent, volatile=True), Var(rsent, volatile=True), Var(ltag, volatile=True), \
                                                     Var(rtag, volatile=True), Var(lrel, volatile=True), Var(rrel, volatile=True)
            target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput, ltag, rtag, lrel, rrel = linput.cuda(), rinput.cuda(), ltag.cuda(), rtag.cuda(), lrel.cuda(), rrel.cuda()
                target = target.cuda()
            lemb, ltagemb, lrelemb = self.embedding_model(linput, ltag, lrel)
            remb, rtagemb, rrelemb = self.embedding_model(rinput, rtag, rrel)
            output = self.model(ltree, lemb, ltagemb, lrelemb, rtree, remb, rtagemb, rrelemb)
            err = self.criterion(output, target)
            loss += err.data[0]
            predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/n_dataset, predictions
