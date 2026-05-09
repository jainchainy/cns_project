import torch
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.train import train_one_epoch_made
from .utils.validation import val_made
import sys
import os
from .predict_epochs import predict_epochs
import re

# train MADE and record the losses during the training process
def main(feat_dir, model_dir, made_dir, TRAIN, DEVICE, MINLOSS):

    # --------- SET PARAMETERS ----------
    model_name = 'made'
    dataset_name = 'myData'
    train_type = TRAIN
    batch_size = 128
    hidden_dims = [512]
    lr = 1e-4
    random_order = False
    patience = 50
    min_loss = int(MINLOSS)
    seed = 290713

    # FORCE CPU
    cuda_device = None

    plot = True
    max_epochs = 2000
    # -----------------------------------

    # Clean made_dir
    for filename in os.listdir(made_dir):
        os.system('rm ' + os.path.join(made_dir, filename))

    # Get dataset
    data = get_data(dataset_name, feat_dir, train_type, train_type)
    train = torch.from_numpy(data.train.x)

    # Data loaders
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)

    # Model
    n_in = data.n_dims
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True, cuda_device=None)

    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Ensure CPU
    model = model.cpu()

    # Save name
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    # Tracking
    epochs_list = []
    train_losses = []
    val_losses = []

    # Early stopping
    i = 0
    max_loss = np.inf

    # Training loop
    for epoch in range(1, max_epochs):
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader, cuda_device)
        val_loss = val_made(model, val_loader, cuda_device)

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save every 10 epochs
        if epoch % 10 == 0:
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, 'epochs_' + save_name)
            )

            predict_epochs(feat_dir, model_dir, made_dir, TRAIN, 'be', DEVICE, epoch)
            predict_epochs(feat_dir, model_dir, made_dir, TRAIN, 'ma', DEVICE, epoch)

        # Early stopping condition
        if val_loss < max_loss and train_loss > min_loss:
            i = 0
            max_loss = val_loss

            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, save_name)
            )
        else:
            i += 1

        if i < patience:
            print(f"Patience counter: {i}/{patience}")
        else:
            print(f"Patience counter: {i}/{patience}\nTerminate training!")
            break