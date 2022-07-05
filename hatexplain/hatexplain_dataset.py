import os
import numpy as np
from keras.preprocessing import sequence
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as PyG_Dataset
import utils
import json
try:
    import cPickle as pkl
except:
    import pickle as pkl
MAX_LEN = 175


class HateXplainDataset(Dataset):
    def __init__(self, root="data", setting="train", split_val=0.05, model="original"):
        self.class_encoder = {'normal': 0, 'offensive': 1, 'hatespeech': 2}
        # Opening JSON files
        with open(root + '/post_id_divisions.json', 'rb') as f:
            file_names = json.load(f)
        with open(root + '/dataset.json', 'rb') as f:
            dataset = json.load(f)
        self.word_to_id = self.build_vocabulary(file_names, dataset)
        self.id_to_word = {value: key for key, value in self.word_to_id.items()}
        files = os.listdir(root)
        essential_files= ['idx_to_file.pkl', 'id_to_word.pkl', 'rationales_test.npy', 'rationales_train.npy', 'word_to_id.pkl',
               'x_test.npy', 'x_train.npy', 'y_test.npy', 'y_train.npy']
        if all(map(lambda v: v in files, essential_files)):
            if setting == 'train':
                x_train, y_train, rationales_train = np.load(os.path.join(root, 'x_train.npy')), \
                                                     np.load(os.path.join(root, 'y_train.npy')), \
                                                     np.load(os.path.join(root, 'rationales_train.npy'))
                self.data = x_train
                self.targets = y_train
                if model == "l2x":
                    assert 'pred_train.npy' in os.listdir(root), \
                        "File 'pred_train.npy' is not in the 'data' folder. Generate prediction first. "
                    self.targets = np.load('data/pred_train.npy')
                self.rationales = rationales_train
            elif setting == 'test':
                x_test, y_test, rationales_test = np.load(os.path.join(root, 'x_test.npy')), \
                                                     np.load(os.path.join(root, 'y_test.npy')), \
                                                     np.load(os.path.join(root, 'rationales_test.npy'))
                self.data = x_test
                self.targets = y_test
                if model == "l2x":
                    assert 'pred_test.npy' in os.listdir(root), \
                        "File 'pred_train.npy' is not in the 'data' folder. Generate prediction first. "
                    self.targets = np.load('data/pred_test.npy')
                self.rationales = rationales_test
            else:
                raise TypeError("Specified <setting> is not allowed.")

        else:  # process dataset if files not in root directory
            x_train, y_train, rationales_train, \
            x_test, y_test, rationales_test = self.process_dataset(dataset, file_names)
            if setting == 'train':
                self.data = x_train
                self.targets = y_train
                self.rationales = rationales_train
            else:
                self.data = x_test
                self.targets = y_test
                self.rationales = rationales_test

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

    def get_rationale(self, idx):
        masked_sentence = []
        sentence = self.get_text(idx)
        masked_instance = self.data[idx] * self.rationales[idx]
        for x, id in enumerate(masked_instance):
            if id == 0:
                masked_sentence.append("-" * len(sentence.split(" ")[x]))
            else:
                masked_sentence.append(self.id_to_word[id])
        return " ".join(masked_sentence)

    def build_vocabulary(self, file_names, json_data):
        # Instanciate Dict
        word_to_id = {}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        num_words = 3
        # Loop through documents
        all_file_names = file_names['train'] + file_names['test']
        for f_name in all_file_names:
            tokens = json_data[f_name]['post_tokens']
            for word in tokens:
                # Add new words to vocabulary with respective index
                if word not in word_to_id and word.isalnum() and word:
                    word_to_id[word] = num_words
                    num_words += 1
        return word_to_id

    def process_dataset(self, dataset, file_names):
        idx_to_file = {"train": {}, "test": {}}
        # Train/Test Sequences
        train_size, test_size = len(file_names['train']), len(file_names['test'])
        x_train, y_train, rationales_train = np.zeros((train_size, MAX_LEN)), np.zeros((train_size, 3)), np.zeros(
            (train_size, MAX_LEN))
        x_test, y_test, rationales_test = np.zeros((test_size, MAX_LEN)), np.zeros((test_size, 3)), np.zeros(
            (test_size, MAX_LEN))
        # Train Set - Pad and Encode Sequences
        for idx, file_id in enumerate(file_names['train']):
            idx_to_file['train'][idx] = file_id  # idx_to_file mapping
            encoded_sequence = []
            doc = dataset[file_id]['post_tokens']
            rationale = dataset[file_id]['rationales']
            # Select label based on majority from annotators
            labels = np.zeros((1, 3))
            for annotator in dataset[file_id]['annotators']:
                class_id = self.class_encoder[annotator['label']]
                labels[0][class_id] += 1
            labels_idx = labels.argmax(axis=1)
            labels = (labels_idx[:, None] == np.arange(labels.shape[1])).astype(int)
            for word in doc:
                if word not in self.word_to_id:
                    encoded_sequence.append(self.word_to_id['<UNK>'])
                else:
                    encoded_sequence.append(self.word_to_id[word])
            # Pad
            encoded_sequence = sequence.pad_sequences([encoded_sequence], maxlen=175)
            rationale = sequence.pad_sequences(rationale, maxlen=175)
            rationale = rationale.sum(axis=0).clip(0, 1)  # Aggregate rationales (union of rationales)
            encoded_sequence[0][encoded_sequence[0].nonzero()[0][0] - 1] = 1  # Start of sequence
            # Add to dataset arrays
            x_train[idx] = encoded_sequence
            y_train[idx] = labels
            rationales_train[idx] = rationale
        # Test Set - Pad and Encode Sequences
        for idx, file_id in enumerate(file_names['test']):
            idx_to_file['test'][idx] = file_id  # idx_to_file mapping
            encoded_sequence = []
            doc = dataset[file_id]['post_tokens']
            rationale = dataset[file_id]['rationales']
            # Select label based on majority from annotators
            labels = np.zeros((1, 3))
            for annotator in dataset[file_id]['annotators']:
                class_id = self.class_encoder[annotator['label']]
                labels[0][class_id] += 1
            labels_idx = labels.argmax(axis=1)
            labels = (labels_idx[:, None] == np.arange(labels.shape[1])).astype(int)
            for word in doc:
                if word not in self.word_to_id:
                    encoded_sequence.append(self.word_to_id['<UNK>'])
                else:
                    encoded_sequence.append(self.word_to_id[word])
            # Pad
            encoded_sequence = sequence.pad_sequences([encoded_sequence], maxlen=175)
            rationale = sequence.pad_sequences(rationale, maxlen=175)
            rationale = rationale.sum(axis=0).clip(0, 1)  # Aggregate rationales (union of rationales)
            encoded_sequence[0][encoded_sequence[0].nonzero()[0][0] - 1] = 1  # Start of sequence
            # Add to dataset arrays
            x_test[idx] = encoded_sequence
            y_test[idx] = labels
            rationales_test[idx] = rationale

        # Save x_train, y_train, rationales_train
        np.save('x_train.npy', x_train), np.save('y_train.npy', y_train), np.save('rationales_train.npy',
                                                                                  rationales_train)
        # Save x_test, y_test, rationales_test
        np.save('x_test.npy', x_test), np.save('y_test.npy', y_test), np.save('rationales_test.npy', rationales_test)
        # Save word_to_id, id_to_word, dataset idx_to_file
        with open('id_to_word.pkl', 'wb') as f:
            pkl.dump(self.id_to_word, f)
        with open('word_to_id.pkl', 'wb') as f:
            pkl.dump(self.word_to_id, f)
        with open('idx_to_file.pkl', 'wb') as f:
            pkl.dump(idx_to_file, f)
        return x_train, y_train, rationales_train, x_test, y_test, rationales_test


class Graph_HateXplainDataset(PyG_Dataset):
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
        self.root = root
        self.length = 0
        if setting == "train":
            self.data = np.load(root + '/x_train.npy')
            y_train = np.load(root + '/pred_train.npy')
            self.targets = (y_train == y_train.max(axis=1)[:, None]).astype(float)  # Make predictions binary
            self.rationales = np.load(os.path.join(root, 'rationales_train.npy'))
        else:
            self.data = np.load(root + '/x_test.npy')
            y_test = np.load(root + '/pred_test.npy')
            self.targets = (y_test == y_test.max(axis=1)[:, None]).astype(float)  # Make predictions binary
            self.rationales = np.load(os.path.join(root, 'rationales_test.npy'))
        with open(root + '/id_to_word.pkl', 'rb') as f:
            self.id_to_word = pkl.load(f)
        super(Graph_HateXplainDataset, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        """
        If the files exists in the raw_dir, the download method is not triggered. Otherwise it calls download method.
        """
        return ['x_train.npy', 'x_test.npy', 'y_train.npy', 'y_test.npy', 'id_to_word.pkl']

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
        x_train = np.load(self.root + '/x_train.npy')
        y_train = np.load(self.root + '/pred_train.npy')
        y_train = (y_train == y_train.max(axis=1)[:, None]).astype(float) # Make predictions binary
        x_test = np.load(self.root + '/x_test.npy')
        y_test = np.load(self.root + '/pred_test.npy')
        y_test = (y_test == y_test.max(axis=1)[:, None]).astype(float)  # Make predictions binary
        with open(self.root + '/id_to_word.pkl', 'rb') as f:
            self.id_to_word = pkl.load(f)
        # Load Glove embeddings
        glove = utils.load_glove_model()
        if self.setting == 'train':
            # Loop through train documents
            idx = 0
            for i, doc in enumerate(x_train):
                # Build Graph Structure, Node Features, Edge Indices and Edge Features
                graph = utils.build_text_dependency_graph(doc, y_train[i], self.id_to_word, glove)
                # Save Data
                torch.save(graph, os.path.join(self.processed_dir + '/train', f'data_{idx}.pt'))
                idx += 1
            print("Train data was parsed and saved successfully.")
        elif self.setting == 'test':
            # Loop through test documents
            idx = 0
            for i, doc in enumerate(x_test):
                # Build Graph Structure, Node Features, Edge Indices and Edge Features
                graph = utils.build_text_dependency_graph(doc, y_test[i], self.id_to_word, glove)
                # Save Data
                torch.save(graph, os.path.join(self.processed_dir + '/test', f'data_{idx}.pt'))
                idx += 1
            print("Test data was parsed and saved successfully.")
        return True

    def len(self):
        return self.length

    def get(self, idx):
        folder = os.path.join(self.processed_dir, self.setting)
        data = torch.load(os.path.join(folder, f'data_{idx}.pt'))
        return {"input": data, "target": data.y}

    def get_text(self, idx):
        sentence = []
        for num in self.data[idx]:
            if num == 0:
                sentence.append("-")
            else:
                sentence.append(self.id_to_word[num])
        return " ".join(sentence)

    def get_rationale(self, idx):
        masked_sentence = []
        sentence = self.get_text(idx)
        masked_instance = self.data[idx] * self.rationales[idx]
        for x, id in enumerate(masked_instance):
            if id == 0:
                masked_sentence.append("-"*len(sentence.split(" ")[x]))
            else:
                masked_sentence.append(self.id_to_word[id])
        return " ".join(masked_sentence)





if __name__ == "__main__":
    training_data = HateXplainDataset(root="data")
    print("Wait...")
