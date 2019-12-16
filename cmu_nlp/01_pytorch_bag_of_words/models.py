import torch
import torch.nn as nn
from torch.autograd import Variable

class BoW(nn.Module):
    def __init__(self, n_words, n_tags):
        super(BoW, self).__init__()
        """variables"""
        self.bias = Variable(torch.zeros(n_tags), requires_grad=True).type(torch.FloatTensor)
        self.embedding = nn.Embedding(n_words, n_tags)

        #initialise embeddings
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, words):
        emb_out = self.embedding(words)
        out = torch.sum(emb_out, dim=0) + self.bias  # size(out) = N
        out = out.view(1, -1)   # size(out) = 1 x N
        return out
    
class CBoW(nn.Module):
    def __init__(self, nwords, ntags, emb_size):
        super(CBoW, self).__init__()
        self.embedding = nn.Embedding(nwords, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.linear = nn.Linear(emb_size, ntags)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, nwords):
        emb_out = self.embedding(nwords)
        emb_sum = torch.sum(emb_out, dim=0)
        out = emb_sum.view(1, -1) # size(emb_sum) = 1 x emb_size
        out = self.linear(out)
        return out
