import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants

class EmbeddingModel(nn.Module):
    def __init__(self, cuda, vocab_size, tag_vocabsize, rel_vocabsize, word_dim, tag_dim, rel_dim):
        super(EmbeddingModel, self).__init__()
        self.cudaFlag = cuda
        self.word_embedding = nn.Embedding(vocab_size, word_dim,
                                padding_idx=Constants.PAD)
        # embedding for postag and rel
        self.tag_emb = nn.Embedding(tag_vocabsize, tag_dim)
        self.rel_emb = nn.Embedding(rel_vocabsize, rel_dim)

        if self.cudaFlag:
            self.word_embedding = self.word_embedding.cuda()
            self.tag_emb = self.tag_emb.cuda()
            self.rel_emb = self.rel_emb.cuda()

    def forward(self, sent_inputs, tag_inputs, rel_inputs):
        sent_emb = F.torch.unsqueeze(self.word_embedding.forward(sent_inputs), 1)
        tag_emb = F.torch.unsqueeze(self.tag_emb.forward(tag_inputs), 1)
        rel_emb = F.torch.unsqueeze(self.rel_emb.forward(rel_inputs), 1)
        return sent_emb, tag_emb, rel_emb