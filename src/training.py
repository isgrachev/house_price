# -*- coding: utf-8 -*-
"""HW2_Grachev.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Vg_Y9oycQYy4A0gQU05JMfE62OkKbZAH

# Kaggle username: isgrachev (Ilia Grachev)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone

# !pip install catboost
from catboost import CatBoostRegressor, Pool
# !pip install torch
import torch
from torch import nn
from torch.utils.data import TensorDataset

# comment following, if importing local files
from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv('drive/My Drive/hse/DLDT/train.csv')
test = pd.read_csv('drive/My Drive/hse/DLDT/test.csv')

train.head(5)

test.head(5)

"""### Data preprocessing:
Train/validation subset split, missing values imputation

After lookin at the data, it is obvious, that YrBlt columns are irrelevant. What matters is age at year of sale.
Therefore, lets introduce numeric variables '...Age' for all '...YrBlt' variables:
* YearBuilt
* YearRemodAdd
* GarageYrBlt,
by substracting '...YrBlt' from 'YrSold'.
"""

age_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
new_names = ['HouseAge', 'RemodAge', 'GarageAge']

def transform_age(df, age_cols, new_names, remove_yrsold=False):
    for col, name in zip(age_cols, new_names):
        df[name] = df['YrSold'] - df[col]
        df.drop(columns=[col], inplace=True)
    if remove_yrsold:
        df.drop(columns=['YrSold'], inplace=True)
    return df

train = transform_age(train, age_cols, new_names, remove_yrsold=True)
test = transform_age(test, age_cols, new_names, remove_yrsold=True)
test_id = test
X_cols = list(set(train.columns[1:]) - set(['SalePrice']))
X = train[X_cols]
y = train.SalePrice
X_test = test[test.columns[1:]]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)

# separate numerical and categorical columns
cat_cols = []
num_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'O': # object
        cat_cols.append(col)
    else:
        num_cols.append(col)

print(f'Length of train dataset: {X_train.shape[0]}')
print('Amount of missing values among features:')
missing = X_train.isna().sum(axis=0).sort_values(ascending=False)
missing[missing > 0]

"""Lets see if some of the missing values can be safely imputed or some features can be safely dropped.
* PoolQC: NA = no pool
* MiscFeature: NA = no misc. features
* Alley: NA = no alley
* Fence: NA = no fence
* MasVnrType: NA = no masonry veneer (why is area missing values so much smaller?)
* FireplaceQu: NA = no fireplace
* LotFrontage: Linear feet of street connected to property. Can impute medians by LotConfig and Condition1, Condition2 groups
* GarageCond, GarageType, GarageYrBlt, GarageQual, GarageFinish: NA = no garage
* BsmtFinType2, BsmtFinType1: NA = no basement or 1 type only (BsmtFinType1)
* BsmntExposure, BsmntCond, BsmntQual: NA = no basement
* MasVnrArea: impute from medians by same MasVnrType. 0 for MasVnrType == NA, therefore such big difference in missing values

Therefore, only in LotFrontage, MasVnrArea  and Electrical should missing values be imputed. The rest should be made into separate category.
"""

# I want to process fake missing data and potential true missing data,
# like in 'Electrical', separately (for cat columns)
# I can use sets, because the columns will be called in the order of the set, so the columns in new df will be reordered and thats it
# missing_cat = missing[missing > 0].index[missing[missing > 0].index.isin(cat_cols)]
false_missing_cat = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                     'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
                     'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
                     'MiscFeature']
other_cat = list(set(cat_cols) - set(false_missing_cat))
assert len(cat_cols) == len(other_cat) + len(false_missing_cat)
assert X_train.shape[1] == len(other_cat) + len(false_missing_cat) + len(num_cols)

# Perhaps LotFrontage could be imputed with more complex algo, but I want to pipeline it
na_imputation = ColumnTransformer(transformers=[
    # fake Na values in categorical columns are imputed with 'NONE' to destinguish category
    ('cat_fake_na', SimpleImputer(strategy='constant', fill_value='NONE'), false_missing_cat),
    # real categorical NA are imputed with most common
    ('cat_na', SimpleImputer(strategy='most_frequent'), other_cat),
    # Real missing data is imputed with median values.
    ('num_na', SimpleImputer(strategy='median'), num_cols)
    ])

X_train = na_imputation.fit_transform(X_train)
X_val = na_imputation.transform(X_val)
X_test = na_imputation.transform(X_test)

# for EDA
X_train_df = pd.DataFrame(data=X_train, columns=(false_missing_cat + other_cat + num_cols))

"""### EDA:"""

plt.figure(figsize=(4, 2))
plt.hist(y_train)
plt.title('Unscaled target variable')
plt.show()

# it is already clear from the metric that it is better to log-scale the target.
# It is a good idea anyway, but will have to reverse-transform predictions later with exp^log
# It doesn't really matter for trees, but may for neural networks
y_train_scaled, y_val_scaled = np.log(y_train), np.log(y_val)
plt.figure(figsize=(4, 2))
plt.hist(y_train_scaled)
plt.title('Log-scaled target variable')
plt.show()

# Look for high correlation in numeric features:
# it will not strongly affect boosting (although not desirable still, as may
# affect stability of trees) but will affect neural networks if I end up doing them
corr = X_train_df[num_cols].corr()
plt.figure(figsize=(14, 11))
sns.heatmap(corr)

"""No strong correlation in data, although garage area and year of garage built are correlated; as well as age of garage, home and remodeling; garage area and number of cars that fits in garage, maybe some other features. Nothing Decision Trees cant chew through..."""

len(num_cols)

# plt.figure(figsize=(10, 10))
# plt.title('Pairplots for numerical features')
for l in range(0, len(num_cols), 6):
    sns.pairplot(train, x_vars=num_cols[l:(l + 6)], y_vars=['SalePrice'])
    plt.show()

"""We can observe on the plot above that LotArea does not determine the price of the estate, contravercial to common belief. However, there is a strong pattern in price based on 1st and 2nd floor area, as well as above grade (ground) living area square feet GrLivArea, OverallQual.\
There can be surprisingly observed certain negative nonlinear pattern in prices based on house age and garage age. Perhaps, old houses are of better quality
"""

len(cat_cols)

for i in range(10):
    fig, axs = plt.subplots(1, 4, figsize=(11, 11))
    for ax, col in zip(axs, cat_cols[(i * 4):((i + 1) * 4)]):
        counts = X_train_df[col].value_counts(normalize=True)
        ax.set_title(col)
        ax.pie(counts.values, labels=counts.index, autopct='%.2f')
    plt.show()

fig, axs = plt.subplots(1, 3, figsize=(11, 11))
for ax, col in zip(axs, cat_cols[-3:]):
    counts = X_train_df[col].value_counts(normalize=True)
    ax.set_title(col)
    ax.pie(counts.values, labels=counts.index, autopct='%.2f')
plt.show()

"""Overall, numerical variables are not highly correlated and indicate some patterns in the data. A lot of categorical featuares are practically invariant,so it is unlikely that they can be used to achieve significant dataset splits, although may be used to split datasets on lower levels of the tree.

Numeric features are not scaled because decision trees are used as base model. Categorical features are not encoded as catboost will be used for convinience.Target is not scaled, because it is indifferent for the tree.

### Gradient Boosting on decision trees

Loss function is RMSE, may use log-scaled targets for cross validation as such is the tagrget metric on kaggle.
"""

# order of categorical columns after dataset transformation with pipeline:
trans_cat = false_missing_cat + other_cat

# pools for boosting
train_pool = Pool(X_train,
                  # use log-scaled target for kaggle metric
                  np.log(y_train),
                  # categorcial featiures
                  cat_features=list(range(len(trans_cat))))
val_pool = Pool(X_val,
                np.log(y_val),
                cat_features=list(range(len(trans_cat))))
test_pool = Pool(X_test,
                 cat_features=list(range(len(trans_cat))))

# fit untuned model, look at convergence
# our real test is uploaded to kaggle, so we can use validation dataset to compare models without dataleak

untuned_cat = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=3,
                                loss_function='RMSE', random_seed=125, verbose=True,
                                early_stopping_rounds=20)

untuned_cat.fit(train_pool, eval_set=val_pool, plot=True, verbose=50)
untuned_pred = untuned_cat.predict(val_pool)
untuned_score = root_mean_squared_error(np.log(y_val), untuned_pred)

# overfit, although train score converges
print(untuned_score)

def plot_feature_importance(importances, names, top=10, figsize=(10, 10), title='Most important features'):
    names = np.array(names)
    # print(names)
    importances = np.array(importances)
    # print(importances)
    untuned_importance = pd.DataFrame(
        {'feature': names,
        'feature_importance': importances}
        ).sort_values(ascending=True, by='feature_importance').iloc[-top:]

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.barh(untuned_importance['feature'], untuned_importance['feature_importance'])
    plt.show()

plot_feature_importance(untuned_cat.feature_importances_, trans_cat + num_cols,
                        title='Top-10 most important features in untuned model')
# untuned_cat.feature_importances_

"""Rational result in terms of feature importance for the most part, although the model is slightly  overfit. Perhaps, some parameter tuning is in order

### WARNING: 4+ HOURS RUNNING TIME. should've used gpu and smaller grid, changed the code after
"""

# tune the model
tuned_cat = clone(untuned_cat)

param_grid = {
    'learning_rate': np.arange(0.001, 0.11, 0.02),
    'depth': [3, 5, 7],
    'l2_leaf_reg': [3, 5, 7]
}

tuning = tuned_cat.grid_search(param_grid, train_pool, partition_random_seed=129,
                               search_by_train_test_split=False, plot=True)
# 149:	loss: 0.3705036	best: 0.1346419 (1)	total: 5h 38m 37s	remaining: 0us

print('Grid search results:\n', tuning['params'])

tuned_pred = tuned_cat.predict(val_pool)
tuned_score = root_mean_squared_error(np.log(y_val), tuned_pred)
print(
    f'''Validation score:
        Before tuning - {untuned_score}
        After tuning - {tuned_score}'''
)

"""I have waited 5 hours and my grid search returned my original baseline boosting. I guess, I got lucky the first time. Nice :)))) Perhaps, learning rate can be increased, but I'm already late with submission, so I will not try anymore"""

plot_feature_importance(tuned_cat.feature_importances_, trans_cat + num_cols,
                        title='Top-10 most important features in tuned model')

"""Same features are important, maybe in slightly different order"""

best_boosting = untuned_cat
y_pred = best_boosting.predict(test_pool)
boosting_submission = test[['Id']]
boosting_submission['SalePrice'] = np.exp(y_pred)
boosting_submission

# boosting_submission.to_csv('drive/My Drive/hse/DLDT/boosting_submission.csv', sep=',', index=False)

"""### Neural network
The dataset is very little for neural networks, but I will build a simple one. The following neural network has 2 hidden leeayers with 2 neurons in each one with a ReLU activation function and a single output. I will use only numeric features and apply normalisation. There is no need to use dataloaders due to dataset size. Very basic model, no early stopping or regularisation inside the model
"""

X_train_nn = X_train[:, len(trans_cat):]
X_val_nn = X_val[:, len(trans_cat):]
X_test_nn = X_test[:, len(trans_cat):]

scaler = StandardScaler()

X_train_nn = scaler.fit_transform(X_train_nn)
X_val_nn = scaler.transform(X_val_nn)
X_test_nn = scaler.transform(X_test_nn)

X_train_nn = torch.tensor(X_train_nn, dtype=torch.float32)
y_train =  torch.tensor(np.array(y_train), dtype=torch.float32)
X_val_nn = torch.tensor(X_val_nn, dtype=torch.float32)
y_val = torch.tensor(np.array(y_val), dtype=torch.float32)
X_test_nn = torch.tensor(X_test_nn, dtype=torch.float32)

train_dataset = TensorDataset(X_train_nn, np.log(y_train))
val_dataset = TensorDataset(X_val_nn, np.log(y_val))
test_dataset = TensorDataset(X_test_nn)
# datasets = {'train': train_dataset, }

model_nn = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(len(num_cols), 2)),
    ('act1', nn.ReLU()),
    ('fc2', nn.Linear(2, 2)),
    ('act2', nn.ReLU()),
    ('outp', nn.Linear(2, 1))
]))

# this solution is copied from https://discuss.pytorch.org/t/rmse-loss-function/16540/2
# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()

#     def forward(self, yhat,y):
#         return torch.sqrt(self.mse(yhat,y))

#     def backward(self):
#         self.mse(yhat,y)

criterion = nn.MSELoss() #RMSELoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fit_nn(model, optimizer, criterion, train_dataset, val_dataset=None,
           max_epochs=100, plot=True, verbose=True, device=None):
    accuracy = {'train': [], 'validation': []}
    verbose = int(verbose)
    # if device is not None:
    #     model.to(device)
    #     for tensor in train_dataset.tensors:
    #         tensor.to(device)
    #     if val_dataset is not None:
    #         for tensor in val_dataset.tensors:
    #             tensor.to(device)
    for epoch in range(max_epochs):
        # first, train the model and update optimizer
        model.train()
        optimizer.zero_grad()
        pred = model(train_dataset.tensors[0])
        loss = criterion(pred, train_dataset.tensors[1])
        loss.backward()
        optimizer.step()
        accuracy['train'].append(np.sqrt(loss.item()))

        # calculate loss on validation dataset
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_dataset.tensors[0])
                val_loss = np.sqrt(criterion(val_pred, val_dataset.tensors[1]).item())
                accuracy['validation'].append(val_loss)

            # verbose in epoch and plot at the end
            if epoch % verbose == 0:
                print(f"Epoch: {epoch}      Train RMSE: {accuracy['train'][epoch]}      Validation RMSE: {accuracy['validation'][epoch]}")
        else:
            if epoch % verbose == 0:
                print(f"Epoch: {epoch}      Train RMSE: {accuracy['train'][epoch]}")

    if plot == True:
        x = list(range(max_epochs))
        plt.figure(figsize=(8, 8))
        plt.plot(x, accuracy['train'], label='Train loss')
        plt.plot(x, accuracy['validation'], label='Validation loss')
        plt.grid(visible=True)
        plt.legend(loc='upper right')
        plt.title('Model convergence in training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

# fit model and predict results
fit_nn(model=model_nn, optimizer=optimizer, criterion=criterion,
       train_dataset=train_dataset, val_dataset=val_dataset,
       max_epochs=2500, device=device, verbose=100)

y_pred_nn = model_nn(test_dataset.tensors[0])
nn_submission = test[['Id']]
nn_submission['SalePrice'] = np.exp(y_pred_nn.detach())

# nn_submission

# nn_submission.to_csv('drive/My Drive/hse/DLDT/nn_submission.csv', sep=',', index=False)

"""The model converges quickly. My guess is that leraning rate should me in the [0.001 and 0.01] interval, now it is too high, based on the graph. Perhaps, if categorical features were used and batch normalisation was used and there were more samples in the dataset and with a different net structure, a better result could be achieved. Still, boosting haas achieved a better result on this dataset."""