import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reproducibility:
torch.manual_seed(10086)
torch.cuda.manual_seed(1)
np.random.seed(10086)
random.seed(10086)
# Set parameters:
max_features = 5000
maxlen = 400
batch_size = 40
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250

class OriginalModel(nn.Module):
    def __init__(self, embed_dim=embedding_dims, vocab_size=max_features, maxlen=maxlen, num_class=2, hidden_dim=hidden_dims):
        super(OriginalModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=filters, kernel_size=kernel_size)
        self.fc1 = nn.Linear(filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        embeddings = self.emb(x)
        embeddings = embeddings.transpose(-2, -1).contiguous()
        z = self.dropout(embeddings)
        z = F.relu(self.conv(z))
        z = F.adaptive_max_pool1d(z, output_size=(1)).view(z.size(0), -1) # This shit found in some sketchy website saved my day
        z = self.fc1(z)
        z = F.relu(self.dropout(z))
        out = self.fc2(z)
        # Remove if using CrossEntropyLoss = LogSoftmax + NLLLoss (PyTorch already applies Softmax in CrossEntropyLoss)
        # Add if using  NNL
        # out = F.log_softmax(z, dim=1)
        return out

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)

if __name__=="__main__":
    model = OriginalModel()
    x = torch.randint(0, 5000, (40, 400))
    print(model(x).shape)
