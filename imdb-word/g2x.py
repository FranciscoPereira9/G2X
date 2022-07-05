from __future__ import absolute_import, division, print_function
import numpy as np
import random
import copy
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, TransformerConv, TopKPooling, \
    max_pool_neighbor_x, avg_pool_neighbor_x
import utils

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


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)


class GNN_ConcatenateLayer(nn.Module):
    def __init__(self):
        super(GNN_ConcatenateLayer, self).__init__()

    def forward(self, inputs):
        global_info, local_info_graph = inputs
        feature_size = global_info.shape[-1]
        # Concat features array
        local_info_graph.x = torch.tile(local_info_graph.x, [1, 2])
        # Append global features to local ones
        batch_graph = 0
        for i in range(len(local_info_graph.ptr) - 1):
            slice = (local_info_graph.ptr[i], local_info_graph.ptr[i + 1])
            # Create global expanded tensor
            glb = torch.tile(global_info[batch_graph], [slice[1] - slice[0], 1])
            # Concatenate features
            local_info_graph.x[slice[0]:slice[1]] = torch.cat((local_info_graph.x[slice[0]:slice[1], :feature_size],
                                                               glb), dim=-1)
            batch_graph += 1
        return local_info_graph


class GNN_GumbelSelector(nn.Module):
    def __init__(self, feature_size):
        super(GNN_GumbelSelector, self).__init__()
        hidden = feature_size*2
        self.dropout = nn.Dropout(p=0.2)
        self.conv0 = GCNConv(in_channels=feature_size, out_channels=hidden)
        self.pool0 = TopKPooling(hidden, ratio=0.75)
        self.conv1 = GCNConv(in_channels=hidden, out_channels=hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.concatenate = GNN_ConcatenateLayer()
        self.conv2 = GCNConv(in_channels=hidden*2, out_channels=hidden)
        self.conv3 = GCNConv(in_channels=hidden, out_channels=1)

    def forward(self, data):
        g = copy.deepcopy(data)
        # Architect
        g.x = F.relu(self.conv0(g.x, g.edge_index))
        # Global Info
        net_new = global_max_pool(g.x, g.batch)  # Reduce Dimensions
        global_info = self.fc1(net_new)
        # Local Info
        local_info = g
        local_info.x = F.relu(self.conv1(local_info.x, local_info.edge_index))
        local_info.x = F.relu(self.conv1(local_info.x, local_info.edge_index))
        # Concatenation
        concatenated = GNN_ConcatenateLayer()([global_info, local_info])
        concatenated.x = self.dropout(concatenated.x)
        concatenated.x = F.relu(self.conv2(concatenated.x, concatenated.edge_index))
        # Logits
        logits_t = self.conv3(concatenated.x, concatenated.edge_index)
        concatenated.x = logits_t
        return concatenated


class GNN_SampleConcrete(nn.Module):
    def __init__(self, tau=1, k=10, train_explainer=False):
        super(GNN_SampleConcrete, self).__init__()
        self.tau = tau
        self.k = k
        self.train_explainer = train_explainer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, g_logits):
        num_nodes = int(g_logits.x.shape[0])
        unif_shape = [num_nodes, self.k * 3]
        uniforms = torch.rand(unif_shape, dtype=torch.float32).to(self.device)
        gumbel = -torch.log(-torch.log(uniforms))
        # Go through batches
        if self.train_explainer:
            samples_list = []
            for i in range(len(g_logits.ptr) - 1):
                slice = (g_logits.ptr[i], g_logits.ptr[i + 1])
                logits = g_logits.x[slice[0]:slice[1]]
                gumb = gumbel[slice[0]: slice[1]]
                # Gumbel-Softmax
                noisy_logits = (gumb + logits) / self.tau  # smaller the more "discrete" it gets (try different ones 1/0.5)
                samples = torch.softmax(noisy_logits, dim=0)
                samples, samples_ids = torch.max(samples, dim=1) # could also do mean
                samples_list.append(samples)
            g_logits.x = torch.cat(samples_list).unsqueeze(dim=1)
            output = g_logits
        else:
            # Discrete Logits
            discrete_logits_list = []
            for i in range(len(g_logits.ptr) - 1):
                slice = (g_logits.ptr[i], g_logits.ptr[i + 1])
                logits = g_logits.x[slice[0]:slice[1]]
                logits = logits.squeeze()
                # Include cases where all words sum up to less than k
                if len(logits) > 10:
                    lower_bound = torch.topk(logits, k=self.k, dim=-1, largest=True, sorted=True)[0][-1]
                else: # include all of them in explanation (?)
                    lower_bound = torch.topk(logits, k=len(logits), dim=-1, largest=True, sorted=True)[0][-1]
                discrete_logits = torch.ge(logits, lower_bound).type('torch.FloatTensor')
                discrete_logits_list.append(discrete_logits)
            g_logits.x = torch.cat(discrete_logits_list).unsqueeze(dim=1)
            # Remove edges from nodes that are not included
            chosen_nodes = torch.where(g_logits.x == 1)[0]
            output = utils.remove_relations(g_logits.to(self.device), chosen_nodes.to(self.device))
        return output


class GNN_Parameterization(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes):
        super(GNN_Parameterization, self).__init__()
        torch.manual_seed(12345)
        self.Multiply = LambdaLayer(lambda x, y: torch.mul(x, y))
        self.conv1 = GCNConv(feature_size, hidden_channels)
        self.topK_pool = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = GCNConv(hidden_channels, 32)
        self.lin = nn.Linear(32, num_classes)

    def forward(self, input_graph, T_graph):
        g = copy.deepcopy(input_graph)
        # Merge inputs and logits
        g.x = self.Multiply(input_graph.x, T_graph.x)
        # 1. Convolution + Pooling
        g.x = self.conv1(g.x, g.edge_index)
        g.x, g.edge_index, _, g.batch, _, _ = self.topK_pool(g.x, g.edge_index, batch=g.batch)
        g.x = F.relu(g.x)
        g.x = self.conv2(g.x, g.edge_index)
        g.x = F.relu(g.x)
        # Readout Layer and Apply Classifier
        out = global_mean_pool(g.x, g.batch)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.lin(out)
        return out


class G2X(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_dims, k, tau=1, train_explainer=False):
        super(G2X, self).__init__()
        self.k = k
        self.tau = tau
        self.Gumbel = GNN_GumbelSelector(feature_size=feature_size)
        self.SampleConcrete = GNN_SampleConcrete(tau=tau, k=k, train_explainer=train_explainer)
        self.QParameterization = GNN_Parameterization(feature_size=feature_size, hidden_channels=hidden_dims,
                                                      num_classes=num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        logits_T = self.Gumbel(x.to(self.device))
        T = self.SampleConcrete(logits_T.to(self.device))
        out = self.QParameterization(x.to(self.device), T.to(self.device))
        return out


class GNN_Explainer(nn.Module):
    def __init__(self, feature_size, k=10):
        super().__init__()
        self.k = k
        self.Gumbel = GNN_GumbelSelector(feature_size=feature_size)
        self.SampleConcrete = GNN_SampleConcrete(k=k, train_explainer=False)

    def forward(self, x):
        logits_T = self.Gumbel(x)
        T = self.SampleConcrete(logits_T)
        return T


class GNN_0(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN_0, self).__init__()
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
                global_representation.append(
                    torch.cat([global_mean_pool(x, batch_index), global_max_pool(x, batch_index)], dim=1))

        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x


def load_pretrained_gnn_gumbel_selector(pretrained_PATH="models/g2x.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create Gumbel Selector
    explainer = GNN_Explainer(feature_size=50)
    explainer_dict = explainer.state_dict()
    # Load pretrained Model
    pretrained_dict = torch.load(pretrained_PATH, map_location=device)
    # Prepare weights transfer
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "Gumbel." in k}
    #pretrained_dict = {k.replace("Gumbel.", ""): v for k, v in pretrained_dict.items()}
    # Overwrite entries in the explainer state dict
    explainer_dict.update(pretrained_dict)
    # Load Explainer's state dict
    explainer.load_state_dict(explainer_dict)
    return explainer


if __name__ == "__main__":
    from imdb_dataset import IMDB_Graph_SentimentDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader

    training_data = IMDB_Graph_SentimentDataset(root="data", setting="train")
    train_loader = PyGDataLoader(training_data, batch_size=2)
    data = next(iter(train_loader))
    print('====DATA===')
    print(data)
    print('===========')
    g2x = G2X(num_classes=2, feature_size=50, hidden_dims=128, k=10, tau=1, train_explainer=True)
    print(g2x(data['input']))
    model = GNN_GumbelSelector(feature_size=50)
    g_logits_T = model(data['input'])
    sample = GNN_SampleConcrete(tau=1, k=10, train_explainer=True)
    T = sample(g_logits_T)
    q = GNN_Parameterization(feature_size=50, hidden_channels=128, num_classes=2)
    out = q(data['input'], T)
    print(out)