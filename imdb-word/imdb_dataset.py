import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as PyG_Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from keras.preprocessing import sequence
from keras.datasets import imdb
import os

try:
    import cPickle as pkl
except:
    import pickle as pkl
import utils

# Reproducibility:
torch.manual_seed(10086)
torch.cuda.manual_seed(1)
np.random.seed(10086)
random.seed(10086)
# Set parameters:
max_features = 5000
maxlen = 400

class IMDB_SentimentDataset(Dataset):

    def __init__(self, data_path="data", setting="train", split_val=0.05, model="original"):
        if 'id_to_word.pkl' not in os.listdir(data_path):
            # Load data from original dataset
            (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features, index_from=3)
            word_to_id = imdb.get_word_index()
            word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
            word_to_id["<PAD>"] = 0
            word_to_id["<START>"] = 1
            word_to_id["<UNK>"] = 2
            id_to_word = {value: key for key, value in word_to_id.items()}
            # Pad Reviews
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
            y_train = np.eye(2)[y_train]
            y_val = np.eye(2)[y_val]
            # Save Data
            np.save(data_path+'/x_train.npy', x_train)
            np.save(data_path+'/y_train.npy', y_train)
            np.save(data_path+'/x_val.npy', x_val)
            np.save(data_path+'/y_val.npy', y_val)
            with open(data_path+'/id_to_word.pkl', 'wb') as f:
                pkl.dump(id_to_word, f)
        assert (setting == "train" or setting == "val" or setting == "test"), \
            "Parameter <setting> must be 'train', 'val' or 'test'."
        if setting == "train":
            self.data = np.load(data_path + '/x_train.npy')
            self.targets = np.load(data_path + '/y_train.npy')
            split = int(len(self.data)*(1-split_val))
            self.data = self.data[:split]
            self.targets = self.targets[:split]
        elif setting == "val":
            self.data = np.load(data_path + '/x_train.npy')
            self.targets = np.load(data_path + '/y_train.npy')
            split = int(len(self.data) * (1-split_val))
            self.data = self.data[split:]
            self.targets = self.targets[split:]
        else:
            self.data = np.load(data_path + '/x_val.npy')
            self.targets = np.load(data_path + '/y_val.npy')

        if model=="l2x":
            if setting=="train":
                assert 'pred_train.npy' in os.listdir(data_path),\
                    "File 'pred_train.npy' is not in the 'data' folder. Generate prediction first. "
                self.targets = np.load('data/pred_train.npy')
            if setting == "test":
                assert 'pred_test.npy' in os.listdir(data_path), \
                    "File 'pred_train.npy' is not in the 'data' folder. Generate prediction first. "
                self.targets = np.load('data/pred_test.npy')

        with open(data_path + '/id_to_word.pkl', 'rb') as f:
            self.id_to_word = pkl.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        target = self.targets[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }

    def get_text(self, idx):
        sentence = []
        for num in self.data[idx]:
            if num == 0:
                sentence.append("-")
            else:
                sentence.append(self.id_to_word[num])
        return " ".join(sentence)



class IMDB_Graph_SentimentDataset(PyG_Dataset):
    def __init__(self, root, setting, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root: where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and
                  processed_dir (processed dataset).
            transform: transformations to be used for data augmentation.
            pre_transform: pre-processing transformations.
            pre_filter: pre_selection of data.
        """
        assert setting in ["train", "test"], "setting must be either 'train' or 'test'."
        self.setting = setting
        self.length = 0
        super(IMDB_Graph_SentimentDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """
        If the files exists in the raw_dir, the download method is not triggered. Otherwise it calls download method.
        """
        return ['x_train.npy', 'x_val.npy', 'y_train.npy', 'y_val.npy', 'id_to_word.pkl']

    @property
    def processed_file_names(self):
        """
        If the files exists in the processed_dir, the process method is not triggered. Otherwise it calls process method.
        """
        files = os.listdir(os.path.join(self.processed_dir, self.setting))
        if len(files) == 0:
            out = ['non_existing_file.pt']
        else:
            self.length = len(files)
            out = [os.path.join(self.setting, file) for file in files]
        return out

    def download(self):
        # Download to `self.raw_dir`. path = download_url(url, self.raw_dir)
        # Not being used.
        pass

    def process(self):
        if 'id_to_word.pkl' not in os.listdir(self.raw_dir):
            # Load data from original dataset
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=3)
            word_to_id = imdb.get_word_index()
            word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
            word_to_id["<PAD>"] = 0
            word_to_id["<START>"] = 1
            word_to_id["<UNK>"] = 2
            self.word_to_id = word_to_id
            id_to_word = {value: key for key, value in word_to_id.items()}
            self.id_to_word = id_to_word
            # Pad Reviews
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
            y_train = np.eye(2)[y_train]
            y_test = np.eye(2)[y_test]
            # Save Data
            np.save(self.raw_dir + '/x_train.npy', x_train)
            np.save(self.raw_dir + '/y_train.npy', y_train)
            np.save(self.raw_dir + '/x_val.npy', x_test)
            np.save(self.raw_dir + '/y_val.npy', y_test)
            with open(self.raw_dir + '/id_to_word.pkl', 'wb') as f:
                pkl.dump(self.id_to_word, f)
        else:
            x_train = np.load(self.raw_dir + '/x_train.npy')
            y_train = np.load(self.raw_dir + '/y_train.npy')
            x_test = np.load(self.raw_dir + '/x_val.npy')
            y_test = np.load(self.raw_dir + '/y_val.npy')
            with open(self.raw_dir + '/id_to_word.pkl', 'rb') as f:
                self.id_to_word = pkl.load(f)
        # Load Glove embeddings
        glove = utils.load_glove_model()
        if self.setting == 'train':
            # Loop through train documents
            idx = 0
            for i, doc in enumerate(x_train):
                # Build Graph Structure, Node Features, Edge Indices and Edge Features
                graph = utils.build_graph(doc, y_train[i], self.id_to_word, glove)
                # Save Data
                torch.save(graph, os.path.join(self.processed_dir + '/train', f'data_{idx}.pt'))
                idx += 1
                #if idx == 500:
                #    break
            print("Train data was parsed and saved successfully.")
        elif self.setting == 'test':
            # Loop through test documents
            idx = 0
            for i, doc in enumerate(x_test):
                # Build Graph Structure, Node Features, Edge Indices and Edge Features
                graph = utils.build_graph(doc, y_test[i], self.id_to_word, glove)
                # Save Data
                torch.save(graph, os.path.join(self.processed_dir + '/test', f'data_{idx}.pt'))
                idx += 1
                #if idx == 500:
                #    break
            print("Test data was parsed and saved successfully.")
        return True

    def len(self):
        return self.length

    def get(self, idx):
        folder = os.path.join(self.processed_dir, self.setting)
        data = torch.load(os.path.join(folder, f'data_{idx}.pt'))
        return {"input": data, "target": data.y}


if __name__ == "__main__":
    training_data = IMDB_Graph_SentimentDataset(root="data", setting="train")
    test_data = IMDB_Graph_SentimentDataset(root="data", setting="test")
    train_loader = PyGDataLoader(training_data, batch_size=2)
    test_loader = PyGDataLoader(test_data, batch_size=2)
    for step, data in enumerate(train_loader):
        print('=======')
        print(f'Step {step + 1}:')
        print(data)
        print()
    print("Wait...")
