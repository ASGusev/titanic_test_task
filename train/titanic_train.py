from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from serialization import save_end_to_end_pipeline


CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']
NUMERIC_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare']


def make_titanic_pipeline(reg_param):
    numeric_preprocessor = Pipeline([
        ('fill', SimpleImputer()),
        ('scale', MinMaxScaler())
    ])
    categorical_preprocessor = Pipeline([
        ('fill', SimpleImputer(strategy='constant', fill_value='na')),
        ('onehot', OneHotEncoder(drop='first'))
    ])
    preprocessor = ColumnTransformer([
        ('numeric', numeric_preprocessor, NUMERIC_FEATURES),
        ('categorical', categorical_preprocessor, CATEGORICAL_FEATURES)
    ])
    logistic_regression = LogisticRegression(penalty='l2', C=1 / reg_param)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('predictor', logistic_regression)
    ])


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_path', default='train.csv',
                            help='Path to the data set. By default is train.csv')
    arg_parser.add_argument('--pipeline_path', default='titanic_pipeline.json',
                            help='Path to save the trained pipeline. By default is titanic_pipeline.json')
    arg_parser.add_argument('--val_share', type=float, default=0.15,
                            help='Share of the data to use for validation. By default is 0.15')
    arg_parser.add_argument('--reg_param', type=float, default=0.5,
                            help='L2 regularization coefficient. By default is 0.5')  # Selected by hand
    args = arg_parser.parse_args()

    data_set = pd.read_csv(args.data_path)
    train_data, val_data = train_test_split(data_set, test_size=args.val_share)

    x_train, y_train = train_data.drop('Survived', axis=1), train_data['Survived']
    pipeline = make_titanic_pipeline(reg_param=args.reg_param).fit(x_train, y_train)
    save_end_to_end_pipeline(pipeline, args.pipeline_path)

    x_val, y_val = val_data.drop('Survived', axis=1), val_data['Survived']
    prediction_train = pipeline.predict(x_train)
    prediction_val = pipeline.predict(x_val)

    print(f'Logistic loss on training data {log_loss(y_train, prediction_train)}')
    print(f'Logistic loss on validation data {log_loss(y_val, prediction_val)}')
