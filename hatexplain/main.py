from original_model import OriginalModel, TextClassificationModel
from hatexplain_dataset import HateXplainDataset, Graph_HateXplainDataset
from helper import train_routine, make_predictions, make_explanations, make_gnn_explanations, \
    make_gnn_soft_explanations, visualise_explanations, evaluate_explanations, make_faithfulness_predictions
from l2x import L2X, load_pretrained_gumbel_selector
from g2x import G2X, load_pretrained_gnn_gumbel_selector
from utils import create_dataset_from_score, create_explanation_dataset, create_soft_explanation_dataset
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
max_features = 26098
maxlen = 175
embedding_dims = 50
filters = 150
kernel_size = 3
hidden_dims = 150


def generate_original_preds(parameters, train=False):
    """
    Generate the predictions of the original model on training and validation datasets.
    Args:
        train: boolean to identify if train is to be performed.
    """
    # Datasets
    my_training_data = HateXplainDataset(root="data", setting="train")
    #my_validation_data = IMDB_SentimentDataset("data", "val")
    my_test_data = HateXplainDataset(root="data", setting="test")
    # Generate PyTorch Dataset/Dataloaders
    train_loader = DataLoader(my_training_data, batch_size=parameters['batch'])
    #val_loader = DataLoader(my_validation_data, batch_size=parameters['batch'])
    test_loader = DataLoader(my_test_data, batch_size=parameters['batch'])
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define model type
    model = OriginalModel(vocab_size=max_features, maxlen=maxlen, num_class=parameters['num_classes'])  # Alternative: model = TextClassificationModel(max_features, embedding_dims, 2)
    if train:
        # Train model from scratch
        train_routine(model, parameters, train_loader, val_loader=None, model_path="models/torch_original.pth")
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


def generate_faithfulness_scores(parameters, explainer, set='test'):
    # Datasets
    my_training_data = HateXplainDataset(root="data", setting="train")
    # my_validation_data = IMDB_SentimentDataset("data", "val")
    my_test_data = HateXplainDataset(root="data", setting="test")
    # Generate PyTorch Dataset/Dataloaders
    train_loader = DataLoader(my_training_data, batch_size=1)
    test_loader = DataLoader(my_test_data, batch_size=1)
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define model type
    model = OriginalModel(vocab_size=max_features, maxlen=maxlen, num_class=parameters[
        'num_classes'])  # Alternative: model = TextClassificationModel(max_features, embedding_dims, 2)
    # Load model from weights
    PATH = "models/torch_original.pth"
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    print("Model loaded.")
    # Predictions on Train Set
    if set == 'train':
        # Generate Comprehensiveness and Sufficiency classification scores
        acc_rationales, sufficiency_predictions, acc_contrast, comprehensiveness_predictions = \
            make_faithfulness_predictions(model, train_loader, f"data/x_train-{explainer}.npy")
        np.save(f"data/sufficiency_train-{explainer}.npy", sufficiency_predictions)
        np.save(f"data/comprehensiveness_train-{explainer}.npy", comprehensiveness_predictions)
        print('Train Acc -  Rationales: {:.4f} '.format(acc_rationales))
        print('Train Acc - ~Rationales: {:.4f} '.format(acc_contrast))
    elif set == 'test':
        # Generate Comprehensiveness and Sufficiency classification scores
        acc_rationales, sufficiency_predictions, acc_contrast, comprehensiveness_predictions = \
            make_faithfulness_predictions(model, test_loader, f"data/x_test-{explainer}.npy")
        np.save(f"data/sufficiency_test-{explainer}.npy", sufficiency_predictions)
        np.save(f"data/comprehensiveness_test-{explainer}.npy", comprehensiveness_predictions)
        print('Test Acc -  Rationales: {:.4f} '.format(acc_rationales))
        print('Test Acc - ~Rationales: {:.4f} '.format(acc_contrast))
    else:
        print("Try other set...")


def l2x_generate_explanations(parameters, train=False):
    print('Loading dataset...')
    # Create Dataset
    myl2x_training_data = HateXplainDataset(root="data", setting="train", model="l2x")
    myl2x_test_data = HateXplainDataset(root="data", setting="test", model="l2x")
    # Create Dataloaders
    train_loader = DataLoader(myl2x_training_data, batch_size=parameters['batch'])
    test_loader = DataLoader(myl2x_test_data, batch_size=1)
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Creating L2X Model...')
    l2x_model = L2X(vocab_size=max_features, embed_dim=embedding_dims, hidden_dims=hidden_dims, k=parameters['k'],
                    tau=1, train_explainer=train)
    # Train
    if train:
        print("Training L2X Model...")
        train_routine(l2x_model, parameters, train_loader, val_loader=None, model_path="models/l2x.pth")

    # Load L2X Model
    print("Loading L2X Model...")
    PATH = "models/l2x.pth"
    l2x_model = L2X(vocab_size=max_features, embed_dim=embedding_dims, hidden_dims=hidden_dims, k=parameters['k'],
                    tau=1, train_explainer=False)
    l2x_model.load_state_dict(torch.load(PATH, map_location=device))
    l2x_model.eval()
    print("L2X Model loaded.")
    # Predictions on Test Set
    accuracy_test, preds_test, targets_test = make_predictions(l2x_model, test_loader)
    print('Test Acc: {:.4f} '.format(accuracy_test))
    # Load Explainer weights and Make predictions
    print("Loading Explainer...")
    explainer = load_pretrained_gumbel_selector()
    print(" Explainer Model Loaded.")
    # Get explainer selections
    print(" Generating Explanations...")
    # Select Nodes
    scores = make_explanations(explainer, test_loader)
    np.save("data/x_test_soft-L2X.npy", scores)
    create_dataset_from_score(myl2x_test_data, scores, parameters['k'])
    print("Explanations saved.")
    print("Visualizing Explanations...")
    explain_data = np.load("data/x_test-L2X.npy")
    rationales = np.load("data/rationales_test.npy")
    targets = np.load("data/y_test.npy")
    predictions = np.load("data/pred_test.npy")
    sentences, explanations = visualise_explanations(myl2x_test_data, explain_data, targets, predictions,
                                                     n_elements=100, gt_mask=True)
    # Explanation metrics
    #metrics = evaluate_explanations(metrics_list=['iou'], predictions=predictions, targets=targets,
    #                                pred_rationales=explain_data.clip(0, 1), gt_rationales=rationales)
    return explain_data


def g2x_generate_explanations(parameters, train=False):
    print('Loading dataset...')
    # Create Dataset
    myl2x_training_data = Graph_HateXplainDataset(root="data", setting="train")
    myl2x_test_data = Graph_HateXplainDataset(root="data", setting="test")
    # Create Dataloaders
    train_loader = PyGDataLoader(myl2x_training_data, batch_size=parameters['batch'])
    test_loader = PyGDataLoader(myl2x_test_data, batch_size=parameters['batch'], shuffle=False)
    # Device at beginning of the script
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Creating G2X Model...')
    # GNN
    gnn_model = G2X(num_classes=parameters['num_classes'], feature_size=50, hidden_dims=parameters['hidden_dims'],
                    k=parameters['k'], tau=parameters['tau'], train_explainer=train)
    # Train
    if train:
        print("Training G2X Model...")
        train_routine(gnn_model, parameters, train_loader, val_loader=None, model_path="models/g2x.pth")

    # Load L2X Model
    print("Loading GNN Model...")
    PATH = "models/g2x.pth"
    g2x_model = G2X(num_classes=parameters['num_classes'], feature_size=50, hidden_dims=parameters['hidden_dims'],
                    k=parameters['k'], tau=parameters['tau'], train_explainer=train)
    g2x_model.load_state_dict(torch.load(PATH, map_location=device))
    print("G2X Model loaded.")
    # Predictions on Test Set
    accuracy_test, preds_test, targets_test = make_predictions(g2x_model, test_loader)
    print('Test Acc: {:.4f} '.format(accuracy_test))
    # Load Explainer weights and Make predictions
    print("Loading Explainer...")
    hard_explainer = load_pretrained_gnn_gumbel_selector(k=parameters['k'], soft=False)
    soft_explainer = load_pretrained_gnn_gumbel_selector(k=parameters['k'], soft=True)
    print(" Explainer Model Loaded.")
    # Get explainer selections
    print(" Generating Hard Explanations...")
    hard_selection = make_gnn_explanations(hard_explainer, test_loader)
    print(" Generating Soft Explanations...")
    # TODO: verify softmask generation
    soft_selection = make_gnn_soft_explanations(soft_explainer, test_loader)
    # Select Words
    create_explanation_dataset(myl2x_test_data.data, hard_selection, path_to_save="data/x_test-G2X.npy")
    create_soft_explanation_dataset(myl2x_test_data.data, soft_selection, path_to_save="data/x_test_soft-G2X.npy")
    print("Explanations saved.")
    print("Visualizing Explanations...")
    explain_data = np.load("data/x_test-G2X.npy")
    rationales = np.load("data/rationales_test.npy")
    targets = np.load("data/y_test.npy")
    predictions = np.load("data/pred_test.npy")
    sentences, explanations = visualise_explanations(myl2x_test_data, explain_data, targets, predictions,
                                                     n_elements=100, gt_mask=True)  # elements to display
    #metrics = evaluate_explanations(metrics_list=['iou'], predictions=predictions, targets=targets,
    #                                pred_rationales=explain_data.clip(0,1), gt_rationales=rationales)
    #ious_array =  iou(explain_data.clip(0, 1), rationales)
    return explain_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml')
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "specified file does not exist."
    # read yaml file with parameters
    with open(args.config_file) as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
    assert config_list['task'] == "original" or config_list['task'] == "l2x" or config_list['task'] == "g2x", \
        "specified task is incorrect."
    if config_list['task'] == 'original':
        generate_original_preds(config_list['original'], train=config_list['train'])
    elif config_list['task'] == 'l2x':
        l2x_generate_explanations(config_list['l2x'], train=config_list['train'])
        generate_faithfulness_scores(config_list['original'], explainer="L2X",set='test')
    else:
        g2x_generate_explanations(config_list['g2x'], train=config_list['train'])
        generate_faithfulness_scores(config_list['original'], explainer="G2X", set='test')
