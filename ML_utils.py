import numpy as np
import matplotlib.pyplot as plt

from voxel import *
from mol_tools import *

import torch
import torch.nn as nn
import random


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
def count_parameters(model):
    s = 0
    for t in model.parameters():
        s += np.prod(t.shape)
    return s

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        # print("CUDA version:", torch.version.cuda)
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
    
@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys = []
    y_hats = []
    
    for names, x, y in loader:
        y_hat = model(x)
        
        ys.append(y.cpu().numpy())
        y_hats.append(y_hat.cpu().numpy())

    ys = np.vstack(ys).reshape(-1)
    y_hats = np.vstack(y_hats).reshape(-1)
    return ys, y_hats

def predict_epochs(net, ml, epochs=1):
    ys = []
    y_hats = []
    for epoch in range(epochs): # for random rotations
        y, y_hat = predict(net, ml)
        ys.append(y)
        y_hats.append(y_hat)
    
    ys = np.hstack(ys)
    y_hats = np.hstack(y_hats)
        
    return ys, y_hats

def plot_predictions(ys, y_hats, alpha=0.2):
    mse = np.mean((ys - y_hats)**2)
    plt.scatter(ys, y_hats, alpha=alpha)
    plt.xlabel("measured")
    plt.ylabel("predicted")
    
    l = min(ys.min(), y_hats.min()) - 1
    u = max(ys.max(), y_hats.max()) + 1
    
    plt.plot([l,u], [l,u], c="red")
    plt.suptitle(f"MSE = {mse:.4f}")
    
    
def fit(epochs, model, train_loader, val_loader, opt, lr, weight_decay, verbose=True):
    t0 = time.time()
    
    optimizer = opt(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss = nn.MSELoss()
    
    train_mse = []
    val_mse = []
    
    for epoch in range(epochs):
        ys = []
        y_hats = []
        
        # Training Phase 
        model.train()
        counter = 0
        total = len(train_loader)
        for names, x, y in train_loader:
            # print(names)
            y_hat = model(x)
            out = loss(y_hat, y)
            out.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            ys.append(y.cpu().detach().numpy())
            y_hats.append(y_hat.cpu().detach().numpy())
            
            counter += 1
            print(f"Epoch {epoch}: Batch {counter}/{total} processed.      ", end="\r")
            
        ys = np.vstack(ys).reshape(-1)
        y_hats = np.vstack(y_hats).reshape(-1)
        mean_loss = np.mean((ys - y_hats)**2)
        train_mse.append(mean_loss)
        
        model.eval()
        ys_val, y_hats_val = predict(model, val_loader)
        mean_loss_val = np.mean((ys_val - y_hats_val)**2)
        val_mse.append(mean_loss_val)
        
        if verbose:
            print(f"Epoch {epoch}: train loss {mean_loss} val loss {mean_loss_val}")
    
    
        
    t1 = time.time()
    
    print(f"Finished in {t1-t0:.4f}s.                               ")

    plt.plot(train_mse, label="train")
    plt.plot(val_mse, label="validation")
    plt.legend()
    plt.show()