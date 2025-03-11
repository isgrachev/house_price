import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from src.models.neural_network import NeuralNetwork


# ---
# load data/create object

with open('files/data_final/features.txt', 'r') as file:
    content = file.readlines()
    n_cat = int(content[0])
    n_num = int(content[1])
    cat_cols = content[2].split(' ')[:n_cat]
    num_cols = content[2].split(' ')[n_cat:]

X_train_scaled = pd.read_csv('files/data_final/X_train_scaled.csv', names=num_cols)
X_val_scaled = pd.read_csv('files/data_final/X_val_scaled.csv', names=num_cols)
y_train = pd.read_csv('files/data_final/y_train.csv', names= ['target'])
y_val = pd.read_csv('files/data_final/y_val.csv', names= ['target'])


# ---
# Gradient Boosting

model = NeuralNetwork(features_in=n_num, max_epochs=500, device='cpu')
model.fit(np.array(X_train_scaled), np.log(np.array(y_train)), np.array(X_val_scaled), np.log(np.array(y_val)))


# ---
# Save objects

if os.path.exists('files/fitted_obj/nn_regressor.pt'):
    conf = input('Warnong! .pt file with such name exists. Do you want to overwrite? y/n')
    assert conf in ['y', 'n'], '\n Input error. y/n input expected'
    if conf in ['y', 'n']:
        os.remove('files/fitted_obj/nn_regressor.pt')
    else:
        raise ValueError('.pt object with specified name alreaddy exists at location')

torch.save(model.state_dict(), 'files/fitted_obj/nn_regressor.pt')
