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
