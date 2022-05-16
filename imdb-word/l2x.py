"""
The code for constructing the original word-CNN is based on
https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, TransformerConv, TopKPooling

# Reproducibility:
torch.manual_seed(10086)
torch.cuda.manual_seed(1)
np.random.seed(10086)
random.seed(10086)
# Set parameters:
max_features = 5000
maxlen = 400
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250


class GumbelSelector(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(GumbelSelector, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.conv0 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=3, padding='same')
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(100, 100)
        self.concatenate = ConcatenateLayer()
        self.conv2 = nn.Conv1d(in_channels=200, out_channels=100, kernel_size=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=1, kernel_size=1, padding='same')

    def forward(self, x):
        embeddings = self.emb(x)
        embeddings = embeddings.transpose(-2, -1).contiguous()  # Input in PyTorch is different from TensorFlow
        dropout = self.dropout(embeddings)
        first_layer = F.relu(self.conv0(dropout))
        # Global Info
        net_new = F.adaptive_max_pool1d(first_layer, output_size=1).squeeze(-1)  # Reduce Dimensions
        global_info = self.fc1(net_new)
        # Local Info
        net = F.relu(self.conv1(first_layer))
        local_info = F.relu(self.conv1(net))
        # Combination
        concatenated = ConcatenateLayer()([global_info, local_info])
        net = self.dropout(concatenated)
        net = F.relu(self.conv2(net))
        logits_t = self.conv3(net)
        return logits_t


class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, inputs):
        input1, input2 = inputs
        input1_expanded = torch.unsqueeze(input1, dim=-1)  # [batchsize, 1, input1_dim]
        dim1_input2 = int(input2.shape[2])
        input1_expanded = torch.tile(input1_expanded, [1, 1, dim1_input2])
        return torch.cat((input1_expanded, input2), dim=-2)


class SampleConcrete(nn.Module):
    def __init__(self, tau, k, train_explainer):
        super(SampleConcrete, self).__init__()
        self.tau = tau
        self.k = k
        self.train_explainer = train_explainer

    def forward(self, logits):
        # logits: [batch_size, 1, maxlen]
        batch_size = int(logits.shape[0])
        d = int(logits.shape[-1])
        unif_shape = [batch_size, self.k, d]
        # Altered uniform
        uniform = torch.rand(unif_shape, dtype=torch.float32)
        gumbel = -torch.log(-torch.log(uniform))
        noisy_logits = (gumbel + logits) / self.tau  # smaller the more "discrete" it gets (try different ones 1/0.5)
        samples = torch.softmax(noisy_logits, dim=-1)
        samples, samples_ids = torch.max(samples, dim=1)
        logits = logits.squeeze(dim=1)
        lower_bounds = torch.topk(logits, k=self.k, dim=-1, largest=True, sorted=True)[0][:, -1]
        thresholds = lower_bounds.unsqueeze(dim=-1)
        discrete_logits = torch.gt(logits, thresholds).type('torch.FloatTensor')
        if self.train_explainer == True:
            output = samples
        else:
            output = discrete_logits
        return torch.unsqueeze(output, -2)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)


class QParameterization(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, k):
        super(QParameterization, self).__init__()
        self.k = k
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.Multiply = LambdaLayer(lambda x, y: torch.mul(x, y))
        self.Mean = LambdaLayer(lambda x: torch.sum(x, dim=-1) / float(k))
        self.fc1 = nn.Linear(embed_dim, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 2)

    def forward(self, x, T):
        embeddings = self.emb(x)
        embeddings = embeddings.transpose(-2, -1).contiguous()
        multiply = self.Multiply(embeddings, T)
        mean = self.Mean(multiply)
        net = F.relu(self.fc1(mean))
        out = self.fc2(net)  # Apply Cross-Entropy Loss which already contains Softmax activation
        return out


class L2X(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, k, tau=1, train_explainer=False):
        super(L2X, self).__init__()
        self.Gumbel = GumbelSelector(vocab_size=vocab_size, embed_dim=embed_dim)
        self.SampleConcrete = SampleConcrete(tau=tau, k=k, train_explainer=train_explainer)
        self.QParameterization = QParameterization(vocab_size=max_features, embed_dim=embedding_dims,
                                                   hidden_dims=hidden_dims, k=k)

    def forward(self, x):
        logits_T = self.Gumbel(x)
        T = self.SampleConcrete(logits_T)
        out = self.QParameterization(x, T)
        return out


class GNN_0(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(50, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 64)
        self.lin = nn.Linear(64, 2)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GNN(torch.nn.Module):
    def __init__(self, feature_size, embedding_size, num_classes):
        super(GNN, self).__init__()
        n_heads = 3
        self.n_layers = 4
        dropout_rate = 0.2
        top_k_ratio = 0.5
        self.top_k_every_n = 1
        dense_neurons = 256

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     beta=True)

        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        # Linear layers
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch_index = data.batch
        # Initial transformation
        x = self.conv1(x, edge_index)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, _, batch_index, _, _ = \
                    self.pooling_layers[int(i / self.top_k_every_n)](x, edge_index, batch=batch_index)
                # Add current representation
                global_representation.append(torch.cat([global_mean_pool(x, batch_index), global_max_pool(x, batch_index)], dim=1))

        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x




def load_pretrained_gumbel_selector(pretrained_PATH="models/l2x.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create Gumbel Selector
    explainer = GumbelSelector(vocab_size=max_features, embed_dim=embedding_dims)
    explainer_dict = explainer.state_dict()
    # Load pretrained Model
    pretrained_dict = torch.load(pretrained_PATH, map_location=device)
    # Prepare weights transfer
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "Gumbel." in k}
    pretrained_dict = {k.replace("Gumbel.", ""): v for k, v in pretrained_dict.items()}
    # Overwrite entries in the explainer state dict
    explainer_dict.update(pretrained_dict)
    # Load Explainer's state dict
    explainer.load_state_dict(explainer_dict)
    return explainer


if __name__ == "__main__":
    x = torch.randint(0, 500, (20, 400))
    k = 10
    model = GumbelSelector(vocab_size=max_features, embed_dim=embedding_dims)
    logits_T = model(x)
    sample = SampleConcrete(tau=1, k=k, train_explainer=False)
    T = sample(logits_T)
    q = QParameterization(vocab_size=max_features, embed_dim=embedding_dims, hidden_dims=hidden_dims, k=k)
    out = q(x, T)
    l, T = L2X(vocab_size=max_features, embed_dim=embedding_dims, hidden_dims=hidden_dims, k=10, tau=1, train_explainer=False)
    print(l(x).shape)