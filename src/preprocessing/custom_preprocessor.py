from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class CustomPreprocessor:
    """
    Custom preprocessor class, that stores imputer and scaler as attributes. They can be fitted to the dataset or used to transform the dataset based on the previous fit.

    Args:
    -----
    cat_cols: list
        List of categorical columns. Needed to separate categorical columns by imputation techniques: most common or 'NONE'
    num_cols: list
        List of numeric columns. Imputed by median value
    """

    def __init__(self, cat_cols: list, num_cols: list):
        # self.data = data
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        # impute missing values:
        false_missing_cat = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
                            'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
                            'MiscFeature']
        other_cat = list(set(cat_cols) - set(false_missing_cat))
        assert len(self.cat_cols) == len(other_cat) + len(false_missing_cat), 'Some categorical colummns are missing'

        self.imputer = ColumnTransformer(transformers=[
            
            # fake Na values in categorical columns are imputed with 'NONE' to destinguish category
            ('cat_fake_na', SimpleImputer(strategy='constant', fill_value='NONE'), false_missing_cat),

            # real categorical NA are imputed with most common
            ('cat_na', SimpleImputer(strategy='most_frequent'), other_cat),

            # Real missing data is imputed with median values.
            ('num_na', SimpleImputer(strategy='median'), num_cols)
            ])
        self.columns = false_missing_cat + other_cat + num_cols
        # std_scale:
        self.scaler = StandardScaler()
        