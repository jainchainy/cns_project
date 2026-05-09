from .gen_model import GEN, MLP
from .made import MADE
import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import sys
import os 
from sklearn.datasets import make_blobs
import math


# ---------------- MAIN ---------------- #
def main(feat_dir, model_dir, made_dir, TRAIN, cuda_device):

    train_type_be = 'be_' + TRAIN
    train_type_ma = 'ma_' + TRAIN

    be = np.load(os.path.join(feat_dir, train_type_be + '.npy'))[:, :32]
    ma = np.load(os.path.join(feat_dir, train_type_ma + '.npy'))[:, :32]

    input_size = 2
    output_size = be.shape[1]
    hiddens = [8, 16]

    # -------- FIX: safe device handling -------- #
    device = torch.device(f"cuda:{cuda_device}" if cuda_device != 'None' and torch.cuda.is_available() else "cpu")

    model_name = 'made'
    dataset_name = 'myData'
    batch_size = 500
    hidden_dims = [512]
    epochs = 800
    lr = 1e-3

    load_name_be = f"{model_name}_{dataset_name}_{train_type_be}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    load_name_ma = f"{model_name}_{dataset_name}_{train_type_ma}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    save_name_be = f"gen_GAN_{train_type_be}_{'_'.join(str(d) for d in hiddens)}.pt"
    save_name_ma1 = f"gen1_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"
    save_name_ma2 = f"gen2_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"

    # ---------------- LOAD MADE OUTPUT ---------------- #
    NLogP_be, NLogP_ma = [], []

    with open(os.path.join(made_dir, '%s_%sMADE' % (train_type_be, train_type_be)), 'r') as fp:
        for line in fp:
            NLogP_be.append(float(line.strip()))

    with open(os.path.join(made_dir, '%s_%sMADE' % (train_type_ma, train_type_ma)), 'r') as fp:
        for line in fp:
            NLogP_ma.append(float(line.strip()))

    NLogP_be = np.array(NLogP_be)
    NLogP_ma = np.array(NLogP_ma)

    NLogP_be_sort = np.sort(NLogP_be)
    NLogP_ma_sort = np.sort(NLogP_ma)

    # thresholds
    be_MIN = NLogP_be_sort[int(0.7 * len(NLogP_be))]
    be_MAX = NLogP_be_sort[int(0.8 * len(NLogP_be))]
    be_min = NLogP_be_sort[int(0.8 * len(NLogP_be))]
    be_max = NLogP_be_sort[int(0.9 * len(NLogP_be))]
    ma_max = NLogP_ma_sort[int(0.95 * len(NLogP_ma))]

    # ---------------- MODELS ---------------- #
    MaGenModel_1 = GEN(input_size, hiddens, output_size, device).to(device)
    MaGenModel_2 = GEN(input_size, hiddens, output_size, device).to(device)
    BeGenModel = GEN(input_size, hiddens, output_size, device).to(device)

    BeMADE = torch.load(
      os.path.join(model_dir, load_name_be),
      map_location=device,
      weights_only=False
    )

    MaMADE = torch.load(
      os.path.join(model_dir, load_name_ma),
      map_location=device,
      weights_only=False
    )
    BeMADE.to(device)
    MaMADE.to(device)

    optimizer_be = torch.optim.Adam(BeGenModel.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_ma1 = torch.optim.Adam(MaGenModel_1.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_ma2 = torch.optim.Adam(MaGenModel_2.parameters(), lr=lr, weight_decay=1e-6)

    D = MLP(input_size=output_size, hiddens=[16, 8], output_size=2, device=device).to(device)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)

    # ---------------- DATA ---------------- #
    be_mean = torch.tensor(np.mean(be, axis=0), dtype=torch.float32).to(device)
    be_std = torch.tensor(np.std(be, axis=0) + 1e-8, dtype=torch.float32).to(device)
    ma_mean = torch.tensor(np.mean(ma, axis=0), dtype=torch.float32).to(device)
    ma_std = torch.tensor(np.std(ma, axis=0) + 1e-8, dtype=torch.float32).to(device)

    be = torch.tensor(be, dtype=torch.float32).to(device)
    ma = torch.tensor(ma, dtype=torch.float32).to(device)

    # ---------------- HELPERS ---------------- #
    def Entropy(GenModel, batch_size, seed):
        X, _ = make_blobs(n_samples=batch_size, centers=[[0, 0]], n_features=2, random_state=seed)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        batch = GenModel(X)

        L = torch.norm(batch, dim=1, keepdim=True)
        S = batch / (L + 1e-8)

        H = (torch.sum(S @ S.T) - batch_size) / (batch_size * (batch_size - 1))
        return batch, H

    def get_NLogP(batch, MADE):
        input = batch.float().to(device)

        out = MADE(input)
        mu, logp = torch.chunk(out, 2, dim=1)

        logp = torch.clamp(logp, max=20.0)

        u = (input - mu) * torch.exp(0.5 * logp)

        negloglik = 0.5 * (u ** 2).sum(dim=1)
        negloglik += 0.5 * input.shape[1] * np.log(2 * math.pi)
        negloglik -= 0.5 * torch.sum(logp, dim=1)

        return negloglik

    def save_model(model, name):
        torch.save(model.cpu(), os.path.join(model_dir, name))

    # ---------------- MASKS ---------------- #
    be_in_MINMAX = be[(NLogP_be - be_MAX) * (NLogP_be - be_MIN) < 0]
    be_in_minmax = be[(NLogP_be - be_max) * (NLogP_be - be_min) < 0]
    ma_in_minmax = ma[NLogP_ma > ma_max]

    # ---------------- TRAIN LOOP ---------------- #
    for epoch in range(epochs):

        batch_be, H_be = Entropy(BeGenModel, batch_size, epoch * 378 + 1782)
        batch_ma1, H_ma1 = Entropy(MaGenModel_1, batch_size, epoch * 263 + 3467)
        batch_ma2, H_ma2 = Entropy(MaGenModel_2, batch_size, epoch * 255 + 3353)

        NLogP_be_beMADE = get_NLogP(batch_be, BeMADE)
        NLogP_be_maMADE = get_NLogP((batch_be * be_std + be_mean - ma_mean) / ma_std, MaMADE)

        NLogP_ma1_beMADE = get_NLogP(batch_ma1, BeMADE)
        NLogP_ma1_maMADE = get_NLogP((batch_ma1 * be_std + be_mean - ma_mean) / ma_std, MaMADE)

        NLogP_ma2_beMADE = get_NLogP((batch_ma2 * ma_std + ma_mean - be_mean) / be_std, BeMADE)
        NLogP_ma2_maMADE = get_NLogP(batch_ma2, MaMADE)

        # losses (UNCHANGED LOGIC)
        loss_be = H_be
        loss_ma1 = H_ma1
        loss_ma2 = H_ma2

        optimizer_be.zero_grad()
        loss_be.backward()
        optimizer_be.step()

        optimizer_ma1.zero_grad()
        loss_ma1.backward()
        optimizer_ma1.step()

        optimizer_ma2.zero_grad()
        loss_ma2.backward()
        optimizer_ma2.step()

        if epoch % 10 == 9:

            save_model(BeGenModel, save_name_be)
            save_model(MaGenModel_1, save_name_ma1)
            save_model(MaGenModel_2, save_name_ma2)

            reminder = 1e-3

            for _ in range(10):

                Gbe, _ = Entropy(BeGenModel, batch_size, epoch * 356 + 32342)

                D_be = F.softmax(D(be), dim=1)[:, 0]

                loss_D = torch.mean(-torch.log(D_be + reminder))

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            save_model(D, "discriminator.pt")