import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# print(sys.path)
# print(os.getcwd())
from src.models.gradient_boosting import CustomCatBoostRegressor


# ---
# load data/create object

with open('files/data_final/features.txt', 'r') as file:
    content = file.readlines()
    n_cat = int(content[0])
    n_num = int(content[1])
    cat_cols = content[2].split(' ')[:n_cat]
    num_cols = content[2].split(' ')[n_cat:]

X_train = pd.read_csv('files/data_final/X_train.csv', names= cat_cols + num_cols)
X_val = pd.read_csv('files/data_final/X_val.csv', names= cat_cols + num_cols)
y_train = pd.read_csv('files/data_final/y_train.csv', names= ['target'])
y_val = pd.read_csv('files/data_final/y_val.csv', names= ['target'])


# ---
# Gradient Boosting

boosting = CustomCatBoostRegressor(random_seed=125, iterations=1000, learning_rate=0.1, depth=3, 
                                   loss_function='RMSE', verbose=True, early_stopping_rounds=20)

boosting.gb_pooling(n_cat, X_train, np.log(y_train), X_val, np.log(y_val))
boosting.fit(boosting.main_pool, eval_set=boosting.val_pool, plot=True, verbose=50)


# ---
# Save objects

if os.path.exists('files/fitted_obj/gb_regressor.cbm'):
    conf = input('Warnong! .cbm file with such name exists. Do you want to overwrite? y/n')
    assert conf in ['y', 'n'], '\n Input error. y/n input expected'
    if conf in ['y', 'n']:
        os.remove('files/fitted_obj/gb_regressor.cbm')
    else:
        raise ValueError('.cbm object with specified name alreaddy exists at location')

boosting.save_model('files/fitted_obj/gb_regressor')
