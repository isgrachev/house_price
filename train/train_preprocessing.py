import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# print(sys.path)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# import src
from src.features import transform_age
from src.preprocessing import CustomPreprocessor


# # ---
# load data/create datasets and objects

dataset = pd.read_csv('files/data_raw/train.csv')
dataset = transform_age(dataset)

X_train, X_val, y_train, y_val = train_test_split(dataset.drop(columns=['Id', 'SalePrice']), dataset['SalePrice'], test_size=0.2, random_state=25)

cat_cols = []
num_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'O': # object
        cat_cols.append(col)
    else:
        num_cols.append(col)

preprocessor = CustomPreprocessor(cat_cols, num_cols)


# ---
# Imputer

X_train = preprocessor.imputer.fit_transform(X_train)
X_val = preprocessor.imputer.transform(X_val)

# save preprocessed datasets
pd.DataFrame(X_train).to_csv('files/data_final/X_train.csv', index=False, header=False)
pd.DataFrame(X_val).to_csv('files/data_final/X_val.csv', index=False, header=False)
pd.DataFrame(y_train).to_csv('files/data_final/y_train.csv', index=False, header=False)
pd.DataFrame(y_val).to_csv('files/data_final/y_val.csv', index=False, header=False)


# ---
# Scaler

X_train_scaled = preprocessor.scaler.fit_transform(X_train[:, len(cat_cols):]) # only numeric fetures for now
X_val_scaled = preprocessor.scaler.transform(X_val[:, len(cat_cols):])

# save preprocessed datasets
pd.DataFrame(X_train_scaled).to_csv('files/data_final/X_train_scaled.csv', index=False, header=False)
pd.DataFrame(X_val_scaled).to_csv('files/data_final/X_val_scaled.csv', index=False, header=False)


# ---
# save objects

# save columns info to txt:
with open('files/data_final/features.txt', 'w') as file:
    file.write(str(len(cat_cols)) + '\n')
    file.write(str(len(num_cols)) + '\n')
    file.write(' '.join(preprocessor.columns))

# save preprocessor
if os.path.exists('files/fitted_obj/preprocessor.joblib'):
    conf = input('Warnong! Joblib file with such name exists. Do you want to overwrite? y/n: ')
    assert conf in ['y', 'n'], '\n Input error. y/n input expected'
    if conf in ['y', 'n']:
        os.remove('files/fitted_obj/preprocessor.joblib')
    else:
        raise ValueError('.joblib object with specified name alreaddy exists at location')

joblib.dump(preprocessor, 'files/fitted_obj/preprocessor.joblib')
