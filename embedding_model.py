import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants

class EmbeddingModel(nn.Module):
    def __init__(self,vocab_size, tag_vocabsize, rel_vocabsize , in_dim,):
        super(EmbeddingModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,in_dim,
                                padding_idx=Constants.PAD)
        # embedding for postag and rel
        self.tag_emb = nn.Embedding(tag_vocabsize, in_dim)
        self.rel_emb = nn.Embedding(rel_vocabsize, in_dim)

    def forward(self, sent_inputs, tag_inputs, rel_inputs):
        sent_emb = F.torch.unsqueeze(self.word_embedding.forward(sent_inputs), 1)
        tag_emb = F.torch.unsqueeze(self.tag_emb.forward(tag_inputs), 1)
        rel_emb = F.torch.unsqueeze(self.rel_emb.forward(rel_inputs), 1)
        return sent_emb, tag_emb, rel_emb