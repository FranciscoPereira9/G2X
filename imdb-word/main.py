from original_model import OriginalModel, TextClassificationModel
from imdb_dataset import IMDB_SentimentDataset, IMDB_Graph_SentimentDataset
from helper import train_routine, make_predictions, make_explanations, visualise_explanations
from l2x import L2X, load_pretrained_gumbel_selector, GNN
from utils import create_dataset_from_score
import random
import os
import numpy as np
import argparse
import yaml
import wandb
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

try:
   import cPickle as pkl
except:
   import pickle as pkl

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


def generate_original_preds(parameters, train=False):
    """
    Generate the predictions of the original model on training and validation datasets.
    Args:
        train: boolean to identify if train is to be performed.
    """
    # Datasets
    my_training_data = IMDB_SentimentDataset("data", "train")
    my_validation_data = IMDB_SentimentDataset("data", "val")
    my_test_data = IMDB_SentimentDataset("data", "test")
    # Generate PyTorch Dataset/Dataloaders
    train_loader = DataLoader(my_training_data, batch_size=parameters['batch'])
    val_loader = DataLoader(my_validation_data, batch_size=parameters['batch'])
    test_loader = DataLoader(my_test_data, batch_size=parameters['batch'])
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define model type
    model = OriginalModel()  # Alternative: model = TextClassificationModel(max_features, embedding_dims, 2)
    if train:
        # Train model from scratch
        train_routine(model, parameters, train_loader, val_loader=val_loader, model_path="models/torch_original.pth")
    else:
        # Load model from weights
        PATH = "models/torch_original.pth"
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval()
        print("Model loaded.")
    # Predictions on Train Set
    accuracy_train, preds_train, targets_train = make_predictions(model, train_loader)
    print('Train Acc: {:.4f} '.format(accuracy_train))
    # Predictions on Test Set
    accuracy_test, preds_test, targets_test = make_predictions(model, test_loader)
    print('Test Acc: {:.4f} '.format(accuracy_test))
    wandb.finish()
    # Save Predictions
    np.save('data/pred_train.npy', preds_train)
    np.save('data/pred_test.npy', preds_test)


def generate_explanations(parameters, train=False):
    print('Loading dataset...')
    # Create Dataset
    myl2x_training_data = IMDB_Graph_SentimentDataset(root="data", setting="train")
    myl2x_test_data = IMDB_Graph_SentimentDataset(root="data", setting="test")
    # Create Dataloaders
    train_loader = DataLoader(myl2x_training_data, batch_size=parameters['batch'])
    test_loader = DataLoader(myl2x_test_data, batch_size=parameters['batch'])
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Creating L2X Model...')
    l2x_model = L2X(vocab_size=max_features, embed_dim=embedding_dims, hidden_dims=hidden_dims, k=parameters['k'], tau=1,
                    train_explainer=True)
    # Train
    if train:
        print("Training L2X Model...")
        train_routine(l2x_model, parameters, train_loader, val_loader=None, model_path="models/l2x.pth")

    # Load L2X Model
    print("Loading L2X Model...")
    PATH = "models/l2x.pth"
    l2x_model = L2X(vocab_size=max_features, embed_dim=embedding_dims, hidden_dims=hidden_dims, k=parameters['k'], tau=1,
                    train_explainer=False)
    l2x_model.load_state_dict(torch.load(PATH, map_location=device))
    l2x_model.eval()
    print("L2X Model loaded.")
    # Load Explainer weights and Make predictions
    print("Loading Explainer...")
    explainer = load_pretrained_gumbel_selector()
    print(" Explainer Model Loaded.")
    # Get explainer selections
    print(" Generating Explanations...")
    scores = make_explanations(explainer, test_loader)
    # Select Words
    create_dataset_from_score(myl2x_test_data, scores, parameters['k'])
    print("Explanations saved.")
    print("Visualizing Explanations...")
    explain_data = np.load("data/x_val-L2X.npy")
    sentences, explanations = visualise_explanations(myl2x_test_data, explain_data, n_elements=parameters['k'])
    return explain_data


def parametrize_predictions(parameters, train=False):
    print('Loading dataset...')
    # Create Dataset
    myl2x_training_data = IMDB_Graph_SentimentDataset(root="data", setting="train")
    myl2x_test_data = IMDB_Graph_SentimentDataset(root="data", setting="test")
    # Create Dataloaders
    train_loader = PyGDataLoader(myl2x_training_data, batch_size=parameters['batch'])
    test_loader = PyGDataLoader(myl2x_test_data, batch_size=parameters['batch'])
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Creating GNN Model...')
    # GNN
    gnn_model = GNN(250)
    # Train
    if train:
        print("Training GNN Model...")
        train_routine(gnn_model, parameters, train_loader, val_loader=None, model_path="models/gnn.pth")

    # Load L2X Model
    print("Loading GNN Model...")
    PATH = "models/gnn.pth"
    gnn_model = GNN(250)
    gnn_model.load_state_dict(torch.load(PATH, map_location=device))
    gnn_model.eval()
    print("GNN Model loaded.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml')
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "specified file does not exist."
    with open(args.config_file) as file:
        # The FullLoader parameter handles the conversion from YAML, scalar values to Python the dictionary format
        config_list = yaml.load(file, Loader=yaml.FullLoader)
    assert config_list['task'] == "original" or config_list['task'] == "g2x", "specified task is incorrect."
    if config_list['task'] == 'original':
        generate_original_preds(config_list['original'], train=config_list['train'])
    else:
        parametrize_predictions(config_list['g2x'], train=config_list['train'])
