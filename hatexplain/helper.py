import random
import numpy as np
from tqdm import tqdm
import os
import time
import wandb
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


# Reproducibility:
torch.manual_seed(10086)
torch.cuda.manual_seed(1)
np.random.seed(10086)
random.seed(10086)


def train_routine(network, parameters, train_loader, val_loader=None, gpus=1, model_path="models/no_name.pth"):
    # Parameters
    epochs, batch_size, optimizer_name, lr = parameters['epochs'], parameters['batch'], \
                                        parameters['optimizer'], parameters['lr']
    # Logging - If you don't want your script to sync to the cloud
    os.environ['WANDB_MODE'] = 'online'
    settings = wandb.Settings(silent=True)
    # WandB config
    if "k" in parameters.keys():
        init = wandb.init(project=parameters["wandb_project"],
                          entity=parameters["wandb_entity"],
                          config= {"epochs": parameters['epochs'],
                                   "batch": parameters['batch'],
                                   "optimizer": parameters['optimizer'],
                                   "lr": parameters['lr'],
                                   "num_classes": parameters['num_classes'],
                                   "k": parameters['k'],
                                   "tau": parameters['tau'],
                                   "hidden_dims": parameters['hidden_dims']
                                   }
                          )
    else:
        init = wandb.init(project=parameters["wandb_project"],
                          entity=parameters["wandb_entity"],
                          config={"epochs": parameters['epochs'],
                                  "batch": parameters['batch'],
                                  "optimizer": parameters['optimizer'],
                                   "lr": parameters['lr'],
                                   "num_classes": parameters['num_classes']
                                  }
                          )


    # create losses (criterion in pytorch)
    criterion = torch.nn.CrossEntropyLoss()

    assert optimizer_name in ['adam', 'sgd'], "specified <optimizer> is not supported. Choose between 'adam' and 'sgd'."
    # create optimizers
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, nesterov=True)

    # Scheduler
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    # load checkpoint if needed/ wanted
    log_interval = 100
    start_n_iter = 0
    start_epoch = 0

    # if we want to run experiment on multiple GPUs we move the models there
    if gpus > 1:
        network = torch.nn.DataParallel(network)
    # Banner
    print(f"Using CUDA: {torch.cuda.is_available()}")
    # Keep track of experiments -> WandB or Tensorboard
    # Or WandB
    wandb.watch(network)

    # Start the main loop
    for epoch in range(start_epoch, epochs):
        torch.set_grad_enabled(True)
        loss_train, acc_train = fit_model(network, epoch, epochs, criterion, optimizer, train_loader)
        print('Epoch: {}  Train Loss: {:.4f}  Train Acc: {:.4f} '.format(epoch, loss_train, acc_train))
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": epoch,
            "Train Loss": loss_train,
            "Train Acc": acc_train,
        })
        if val_loader:
            loss_valid, acc_valid = validate_model(network, criterion, val_loader)
            # Decay Learning Rate
            #scheduler.step(acc_valid)
            print('Epoch: {}  Valid Loss: {:.4f}  Valid Acc: {:.4f} '.format(epoch, loss_valid, acc_valid))
            # Log the loss and accuracy values at the end of each epoch
            wandb.log({
                "Epoch": epoch,
                "Valid Loss": loss_valid,
                "Valid Acc": acc_valid})
        # else:
            #scheduler.step(acc_train)

    torch.save(network.state_dict(), model_path)


def fit_model(network, epoch, epochs, criterion, optimizer, train_loader):
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        network = network.cuda()
    # Set network to train mode
    network.train()
    # Use tqdm for iterating through data
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    start_time = time.time()
    total_accuracy = []
    total_loss = []
    for batch_idx, data in pbar:
        optimizer.zero_grad()
        inp, target = data.values()
        if use_cuda:
            inp = inp.cuda()
            target = target.cuda()
        # Good practice to keep track of preparation time and computation time to find any issues in your dataloader
        prepare_time = start_time - time.time()
        # Forwards Pass
        #print()
        #print(inp.x)
        output = network(inp)
        # Backward Pass
        loss = criterion(output, target)
        loss.backward()
        # Update
        optimizer.step()
        # Compute computation time and *compute_efficiency*
        process_time = start_time - time.time() - prepare_time
        pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
            process_time / (process_time + prepare_time), epoch, epochs))
        start_time = time.time()
        # Measure Loss and Accuracy
        total_loss.append(loss.item())
        total_accuracy.append(accuracy(output, target))
    # Report epoch metrics
    ls = np.array(total_loss).mean()
    acc = np.array(total_accuracy).mean()
    return ls, acc


def validate_model(network, criterion, val_loader):
    network.eval()
    # Validation
    with torch.no_grad():
        # if running on GPU and we want to use cuda move model there
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            network = network.cuda()
        # Use tqdm for iterating through data
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        start_time = time.time()
        total_accuracy = []
        total_loss = []
        for batch_idx, data in pbar:
            inp, target = data.values()
            if use_cuda:
                inp = inp.cuda()
                target = target.cuda()
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()
            # Generate outputs
            output = network(inp)
            # Compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}".format(
                process_time / (process_time + prepare_time)))
            start_time = time.time()
            # Measure Loss and Accuracy
            total_loss.append(criterion(output, target).item())
            total_accuracy.append(accuracy(output, target))
    # Report epoch metrics
    ls = np.array(total_loss).mean()
    acc = np.array(total_accuracy).mean()
    return ls, acc


def make_predictions(network, loader):
    all_predictions = []
    all_targets = []
    # Prepare for Inference
    network.eval()
    with torch.no_grad():
        # if running on GPU and we want to use cuda move model there
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            network = network.cuda()
        # Use tqdm for iterating through data
        pbar = tqdm(enumerate(loader), total=len(loader))
        start_time = time.time()
        total_accuracy = []
        for batch_idx, data in pbar:
            inp, target = data.values()
            if use_cuda:
                inp = inp.cuda()
                target = target.cuda()
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()
            # Generate outputs
            output = network(inp)
            out_probabilities = F.softmax(output, dim=1)
            # Compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}".format(
                process_time / (process_time + prepare_time)))
            start_time = time.time()
            # Measure Accuracy
            total_accuracy.append(accuracy(output, target))
            # Append Predictions
            all_predictions += out_probabilities.detach().to("cpu").tolist()
            all_targets += target.detach().to("cpu").tolist()

    # Report metrics
    acc = np.array(total_accuracy).mean()
    return acc, np.array(all_predictions), np.array(all_targets)


def make_faithfulness_predictions(network, loader, hard_rationales_path):
    hard_rationales = np.load(hard_rationales_path)
    comprehensiveness_predictions = []
    sufficiency_predictions = []
    # Prepare for Inference
    network.eval()
    with torch.no_grad():
        # if running on GPU and we want to use cuda move model there
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            network = network.cuda()
        # Use tqdm for iterating through data
        pbar = tqdm(enumerate(loader), total=len(loader))
        start_time = time.time()
        rat_total_accuracy = []
        contrast_total_accuracy = []
        for batch_idx, data in pbar:
            inp, target = data.values()
            rat_mask = torch.tensor(hard_rationales[batch_idx], dtype=torch.bool)
            rat_inp = inp * rat_mask
            contrast_inp = inp * ~rat_mask
            if use_cuda:
                rat_inp = rat_inp.cuda()
                contrast_inp = contrast_inp.cuda()
                target = target.cuda()
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()
            # Generate outputs
            rat_output = network(rat_inp)
            rat_out_probabilities = F.softmax(rat_output, dim=1)
            contrast_output = network(contrast_inp)
            contrast_out_probabilities = F.softmax(contrast_output, dim=1)
            # Compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}".format(
                process_time / (process_time + prepare_time)))
            start_time = time.time()
            # Measure Accuracy
            rat_total_accuracy.append(accuracy(rat_output, target))
            contrast_total_accuracy.append(accuracy(contrast_output, target))
            # Append Predictions
            sufficiency_predictions += rat_out_probabilities.detach().to("cpu").tolist()
            comprehensiveness_predictions += contrast_out_probabilities.detach().to("cpu").tolist()

    # Report metrics
    acc_rationales = np.array(rat_total_accuracy).mean()
    acc_contrast = np.array(contrast_total_accuracy).mean()
    return acc_rationales, np.array(sufficiency_predictions), acc_contrast, np.array(comprehensiveness_predictions)


def make_explanations(network, loader):
    all_predictions = []
    all_targets = []
    # Prepare for Inference
    network.eval()
    with torch.no_grad():
        # if running on GPU and we want to use cuda move model there
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            network = network.cuda()
        start_time = time.time()
        # Use tqdm for iterating through data
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, data in pbar:
            inp, target = data.values()
            if use_cuda:
                inp = inp.cuda()
                target = target.cuda()
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()
            # Generate outputs
            output = network(inp)
            out_probabilities = F.softmax(output, dim=-1).squeeze()
            # Compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}".format(process_time / (process_time + prepare_time)))
            start_time = time.time()
            # Append Predictions
            all_predictions.append(out_probabilities.detach().to("cpu").tolist())
    scores = np.array(all_predictions)
    return scores


def make_gnn_explanations(network, loader):
    all_predictions = []
    # Prepare for Inference
    network.eval()
    with torch.no_grad():
        # if running on GPU and we want to use cuda move model there
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            network = network.cuda()
        start_time = time.time()
        # Use tqdm for iterating through data
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, data in pbar:
            inp, target = data.values()
            if use_cuda:
                inp = inp.cuda()
                target = target.cuda()
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()
            # Generate outputs
            output = network(inp)
            # Compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}".format(process_time / (process_time + prepare_time)))
            start_time = time.time()
            # Select Explaining Nodes for each instance
            for i in range(len(data['input'].ptr) - 1):
                slice = (data['input'].ptr[i].item(), data['input'].ptr[i + 1].item())
                sgl_out_labels = output.node_labels[slice[0]:slice[1]]
                sgl_out_x = output.x[slice[0]:slice[1]]
                selected_node_labels = sgl_out_labels[sgl_out_x.squeeze() == 1].detach().to("cpu").tolist() # selected_node_labels = output.node_labels[output.x.squeeze() == 1].detach().to("cpu").tolist()
                all_predictions.append(selected_node_labels)
    return all_predictions


def make_gnn_soft_explanations(network, loader):
    all_soft_masks = []
    # Prepare for Inference
    network.eval()
    with torch.no_grad():
        # if running on GPU and we want to use cuda move model there
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            network = network.cuda()
        start_time = time.time()
        # Use tqdm for iterating through data
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, data in pbar:
            inp, target = data.values()
            if use_cuda:
                inp = inp.cuda()
                target = target.cuda()
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()
            # Generate outputs
            output = network(inp)
            # Compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}".format(process_time / (process_time + prepare_time)))
            start_time = time.time()
            # Select Explaining Nodes for each instance
            for i in range(len(data['input'].ptr) - 1):
                slice = (data['input'].ptr[i].item(), data['input'].ptr[i + 1].item())
                sgl_out_labels = output.node_labels[slice[0]:slice[1]].squeeze().detach().to("cpu").tolist()
                sgl_out_x = output.x[slice[0]:slice[1]].squeeze().detach().to("cpu").tolist()
                mapping = {}
                for label, x in zip(sgl_out_labels, sgl_out_x):
                    mapping[label] = x
                all_soft_masks.append(mapping)
    return all_soft_masks


def evaluate_explanations(metrics_list, predictions, targets, pred_rationales, gt_rationales):
    metrics = {}
    # Confusion Matrix predicted\actual -> model performance
    discrete_predictions = np.eye(predictions.shape[1])[predictions.argmax(axis=-1)]
    confusion_matrix = np.dot(discrete_predictions.T, targets)
    print("Models Performance | Confusion Matrix: \n", confusion_matrix)
    # Normal classes have no rationales, thus we make the pred_rationales 0 where the models predicts normal class
    pred_rationales[discrete_predictions[:, 0] == 1] = 0
    # Calculating metrics based only on predicted explanations and gt_explanations (iou threshold of 0.5)
    tp, fp, fn = 0, 0, 0
    for i in range(len(pred_rationales)):
        iou_value = iou(pred_rationales[i], gt_rationales[i])
        if iou_value >= 0.5:
            tp+=1
        else:
            if (pred_rationales[i].sum() == 0) and (gt_rationales[i].sum() >= 0):
                fn+=1
            else:
                fp+=1
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: ", precision, "Recall: ", recall, "F1: ", f1)
    return metrics


def accuracy(predictions, targets):
    """
    Args:
        predictions: tensor with probabilities for each class ([p11,p12],[p21,p22],...)
        targets: tensor with labels for each class ([l11,l12],[l21,l22],...)
    """
    assert len(predictions) == len(targets), "Error: <predictions> must match <targets> size."
    predictions = torch.argmax(predictions, dim=-1)
    targets = torch.argmax(targets, dim=-1)
    correct = (predictions == targets).sum().item()
    total = len(predictions)
    return correct / total


def iou(pred_rationales, gt_rationales):
    """
    Args:
        explanations, rationales : arrays with shape (N, max_len), where values are 0 or 1.
                                    -> 1 means the word is considered for the explanation
                                    -> 0 otherwise
    """
    assert pred_rationales.shape == gt_rationales.shape, "Arrays shape don't match."
    return np.logical_and(pred_rationales, gt_rationales).sum(axis=-1) / \
           (np.logical_or(pred_rationales, gt_rationales).sum(axis=-1) + 1e-6)


def visualise_explanations(dataset, explain_data, targets, predictions, n_elements, gt_mask=False):
    idxs = np.random.randint(0, len(dataset), n_elements)
    sentences = []
    explanations = []
    rationales = []
    targets_list = []
    predictions_list = []
    for i in idxs:
        # Append respective sentence, rationale, target and prediction
        sentence = dataset.get_text(i)
        sentences.append(sentence)
        if gt_mask:
            rationales.append(dataset.get_rationale(i))
        targets_list.append(targets[i])
        predictions_list.append(predictions[i])
        explanation = []
        for x, code in enumerate(explain_data[i]):
            if code == 0:
                explanation.append("-"*len(sentence.split(" ")[x]))
            else:
                explanation.append(dataset.id_to_word[code])
        aux = " ".join(explanation)
        explanations.append(aux)

    for idx in range(len(sentences)):
        print("Sentence    : ", sentences[idx])
        if gt_mask:
            print("Rationale    : ", rationales[idx])
        print("Explanation : ", explanations[idx])
        print("Targ. | Pred.: ", targets_list[idx], " | ", predictions_list[idx])
        print("--------------------------------------")


    return sentences, explanations

