import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import os
import spacy
dependency_parser = spacy.load("en_core_web_sm")

def get_selected_words(x_single, score, id_to_word, k):
    selected_words = {}  # {location: word_id}
    selected = np.argsort(score)[-k:]
    selected_k_hot = np.zeros(len(x_single))
    selected_k_hot[selected] = 1.0
    x_selected = (x_single * selected_k_hot).astype(int)
    return x_selected


def create_dataset_from_score(dataset, scores, k):
    id_to_word = dataset.id_to_word
    x = dataset.data
    new_data = []
    new_texts = []
    for i, x_single in enumerate(x):
        x_selected = get_selected_words(x_single, scores[i], id_to_word, k)
        new_data.append(x_selected)

    np.save('data/x_test-L2X.npy', np.array(new_data))


def create_explanation_dataset(dataset, selected, path_to_save):
    new_data = []
    for i, x_single in enumerate(dataset):
        mask = np.isin(x_single, selected[i]).astype(int)
        x_selected = (x_single * mask).astype(int)
        new_data.append(x_selected)
    np.save(path_to_save, np.array(new_data))


def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


def create_soft_explanation_dataset(dataset, selected, path_to_save):
    new_data = np.zeros(dataset.shape)
    for i, x_single in enumerate(dataset):
        start = np.where(dataset[i]==1)[0][0]
        importance_scores = np.vectorize(selected[i].get)(dataset[i][start + 1:])
        soft_mask = softmax(importance_scores)
        new_data[i][start+1:] = soft_mask
    np.save(path_to_save, np.array(new_data))


def calculate_acc(pred, y):
    return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))


def preprocess_document(document, sentence_spliter='.', word_spliter=' ', punct_mark=','):
    # lowercase all words and remove trailing whitespaces
    document = document.lower().strip()

    # remove unwanted punctuation marks
    for pm in punct_mark:
        document = document.replace(pm, '')

    # get list of sentences which are non-empty
    sentences = [sent for sent in document.split(sentence_spliter) if sent != '']

    # get list of sentences which are lists of words
    document = []
    for sent in sentences:
        words = sent.strip().split(word_spliter)
        document.append(words)

    return document


def get_entities(document, id_to_word, pre_trained_emb):
    """
    Args:
        document: a list of enconded words in a document -> [0, 0, 0, 50, 29, 500, 20, ...]
    Returns:
        node_features: node feature matrix with shape [#nodes,#features]
        unique_words: encoded words that correspond to the nodes -> node_features[109] corresponds to unique_word[109]
    """

    # in our case, node entities are all the unique words
    node_features = []
    unique_words = []
    for i in range(len(document)):
        if document[i] not in unique_words and document[i] not in [0, 1]:
            # Append node features to list
            word = id_to_word[document[i]]
            if word in pre_trained_emb.keys():
                # pre-trained embedding
                embedding_matrix = pre_trained_emb[word].tolist()
            else:
                # randomly initialized uniform embedding [-1,1]
                embedding_matrix = (2 * np.random.random_sample(50) - 1).tolist()
            node_features.append(embedding_matrix)
            # Track document words
            unique_words.append(document[i])
    return torch.tensor(node_features, dtype=torch.float), torch.tensor(unique_words, dtype=torch.int)


def get_relations(document, p=2):
    """
    Args:
        document: a list of encoded words in a document -> [0, 0, 0, 50, 29, 500, 20, ...]
        p: neighbourhood size to consider
    Returns:
        relations: graph connectivity in COO format with shape [2, num_edges]
    """
    # relate each node to neighbour p neighbours
    unique_words = {}
    node_index = 0
    for i in range(len(document)):
        if document[i] not in unique_words.keys() and document[i] not in [0, 1]:
            # Track document words
            unique_words[document[i]] = node_index
            node_index += 1

    relations = []
    for i in range(len(document) - 1):
        if document[i] not in [0, 1]:
            # for every word and the next ones in the sentence
            for j in range(1, p + 1):
                if (i + j) < len(document):
                    # Append both relation in both directions
                    if document[i + j] not in [0, 1]:
                        pair1 = [unique_words[document[i]], unique_words[document[i + j]]]
                        pair2 = [unique_words[document[i + j]], unique_words[document[i]]]
                        # only add unique bigrams
                        if pair1 not in relations:
                            relations.append(pair1)
                        if pair2 not in relations:
                            relations.append(pair2)
    return torch.tensor(relations, dtype=torch.long).t().contiguous()


def remove_relations(graph, keep_nodes):
    """
    Args:
        graph: Graph Data object
        keep_nodes: tensor with node indices to keep
    """
    # Select edges to drop
    bool_arr_0 = torch.isin(graph.edge_index[0], keep_nodes)
    bool_arr_1 = torch.isin(graph.edge_index[1], keep_nodes)
    bool_arr = bool_arr_0 & bool_arr_1
    # Remove edges from source and destination
    new_edges = torch.tensor([graph.edge_index[0][bool_arr].tolist(),  # source
                              graph.edge_index[1][bool_arr].tolist()],  # destination
                             dtype=torch.long)
    graph.edge_index = new_edges
    return graph


def build_graph(doc, y, id_to_word, pre_trained_emb, preprocess=False):
    # preprocess document for standardization
    if preprocess:
        doc = preprocess_document(doc)

    # get graph nodes
    node_features, unique_words = get_entities(doc, id_to_word, pre_trained_emb)

    # get graph edges
    edges = get_relations(doc, p=3)

    # label to tensor
    y = torch.tensor(y, dtype=torch.float32)

    # Create PyTorch Geometric Data object
    G = Data(x=node_features, edge_index=edges, y=y, node_labels=unique_words)

    return G


def build_text_dependency_graph(doc, y, id_to_word, pre_trained_emb, preprocess=False):
    # TODO: has a bug. The dependency parser tokenizes the text which than makes it non compatible to the tokens of the original dataset.
    # FIXED
    # preprocess document for standardization
    if preprocess:
        doc = preprocess_document(doc)

    doc_tokens = [id_to_word[id] for id in doc if id not in [0, 1]]
    og_len = len(doc_tokens)
    doc_text = " ".join(doc_tokens)
    doc_text = dependency_parser(doc_text)
    spacy_len = len(list(doc_text))

    # in our case, node entities are all the unique words
    node_features = []
    relations = []
    relations_attr = []
    tokens = []
    nth_token = 0
    og_to_parsed = {}
    parsed_to_og = {}
    for i in range(len(doc)):
        if doc[i] not in [0, 1]:
            # Check if dependency parser is the same as og. If not, merge with next one till it forms the same word.
            word = id_to_word[doc[i]]
            if word != str(doc_text[nth_token]):
                j = 1
                while word != str(doc_text[nth_token:nth_token+j]):
                    j+=1
                together = []
                for a in range(j):
                    together.append(nth_token + a)
                    parsed_to_og[nth_token + a] = i
                og_to_parsed[i] = together
                nth_token += j
            else:
                # all good
                og_to_parsed[i] = [nth_token]
                parsed_to_og[nth_token] = i
                nth_token+=1


    nth_token = 0
    for i in range(len(doc)):
        if doc[i] == 1:
            offset = i + 1
        if doc[i] not in [0, 1]:
            nth_token = i - offset
            # Append node features to list
            word = id_to_word[doc[i]]
            if word in pre_trained_emb.keys():
                # pre-trained embedding
                embedding_matrix = pre_trained_emb[word].tolist()
            else:
                # randomly initialized uniform embedding [-1,1]
                embedding_matrix = (2 * np.random.random_sample(50) - 1).tolist()
            node_features.append(embedding_matrix)
            # Track document words
            tokens.append(doc[i])
            # Go through doc text tokens that correspond to doc[i]
            for doc_text_token_i in og_to_parsed[i]:
                # For those search their children in the dependency tree
                for child in doc_text[doc_text_token_i].children:
                    # Associate connections to original token doc[i]
                    pair1 = [nth_token, parsed_to_og[child.i]-offset]
                    pair2 = [parsed_to_og[child.i]-offset, nth_token]
                    if pair1 not in relations:
                        relations.append(pair1)
                        relations_attr.append([child.dep_])
                    if pair2 not in relations:
                        relations.append(pair2)
                        relations_attr.append([child.dep_])

    node_features = torch.tensor(node_features, dtype=torch.float)
    tokens = torch.tensor(tokens, dtype=torch.int)
    edges = torch.tensor(relations, dtype=torch.long).t().contiguous()
    # label to tensor
    y = torch.tensor(y, dtype=torch.float32)

    # Create PyTorch Geometric Data object
    G = Data(x=node_features, edge_index=edges, y=y, node_labels=tokens)

    return G


def load_glove_model(dir="glove"):
    print("Loading Glove pre-trained embeddings...")
    glove_model = {}
    with open(os.path.join(dir, "glove.6B.50d.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def plot_graph(G, title=None):
    # Need pygraphviz to run this
    # set figure size
    plt.figure(figsize=(10, 10))

    # define position of nodes in figure
    pos = nx.nx_agraph.graphviz_layout(G)

    # draw nodes and edges
    nx.draw(G, pos=pos, with_labels=True)

    # get edge labels (if any)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # draw edge labels (if any)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # plot the title (if any)
    plt.title(title)

    plt.show()
    return G
