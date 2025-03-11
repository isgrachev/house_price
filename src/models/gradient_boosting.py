from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CustomCatBoostRegressor(CatBoostRegressor):
    """
    Catboost Regressor with pooling method, that creates dataset pools as attributes of the instance (can be ommited in favour of original methods).
    
    Args:
    -----
    See arguments for Yandex CatBoostRegressor
    """

    def __init__(self, 
                 random_seed, iterations: int = 500, learning_rate: float = 0.1, depth: int = 3, 
                 loss_function = 'RMSE', verbose: bool = True, early_stopping_rounds: int = 20):
        
        super().__init__(random_seed=random_seed, iterations=iterations, learning_rate=learning_rate, 
                         depth=depth, loss_function=loss_function, verbose=verbose, early_stopping_rounds=early_stopping_rounds)

    def gb_pooling(self, cat_cols: int, X, y = None, X_val = None, y_val = None):
        """
        Create data pools for a subsample. 
        Creates pools for: training - if X and y are specified, validation - if X_val and y_val are specified and test 
        """

        # transformed_cat = false_missing_cat + other_cat  # order of categorical columns after dataset transformation with pipeline:
        self.main_pool = Pool(X, y, cat_features=list(range(cat_cols)))
        try:
            self.val_pool = Pool(X_val, y_val, cat_features=list(range(cat_cols)))
        except:
            pass
            

    def plot_feature_importance(self, names, top=10, figsize=(10, 10), title='Most important features'):
        """
        Plot feature importances based on parent class attribute feature_importances_ using matplotlib pyplot.
        
        Args:
        -----
        names: [str]
            Names of variables. Must be in the same order as they were passed into regressor
        top: int, default = 10
            Returns top-n most important features
        figsize: (int, int), default = (10, 10)
            A tuple passed to matplotlib.pyplot.figure figsize parameter
        title: str, default = 'Most important features'
            A string passed to matplotlib.pyplot.title method
        """

        names = np.array(names)
        importances = np.array(self.feature_importances_)
        untuned_importance = pd.DataFrame(
            {'feature': names,
            'feature_importance': importances}
            ).sort_values(ascending=True, by='feature_importance').iloc[-top:]

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.barh(untuned_importance['feature'], untuned_importance['feature_importance'])
        plt.show()
