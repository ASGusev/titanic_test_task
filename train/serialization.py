import json

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def representation_imputer(imputer):
    return {
        'type': 'SimpleImputer',
        'values': [
            {
                'type': 'StringValue' if isinstance(i, str) else 'FloatValue',
                'value': i
            }
            for i in imputer.statistics_
        ]
    }


def representation_scaler(scaler):
    return {
        'type': 'Scale',
        'min': scaler.data_min_.tolist(),
        'range': scaler.data_range_.tolist()
    }


def representation_one_hot(one_hot):
    return {
        'type': 'OneHot',
        'options': [i.tolist() for i in one_hot.categories_]
    }


def representation_transformers_pipeline(transformers_pipeline):
    return {
        'type': 'TransformerPipeline',
        'steps': [make_representation(step) for name, step in transformers_pipeline.steps]
    }


def representation_column_transformer(column_transformer):
    col_names = []
    transformers = []
    for name, columns_transformer, columns in column_transformer.transformers_:
        if name != 'remainder':
            col_names.append(columns)
            transformers.append(make_representation(columns_transformer))
    return {
        'transformers': transformers,
        'colNames': col_names
    }


def representation_logistic_regression(model):
    return {
        'type': 'LogisticRegression',
        'coefficients': list(model.coef_.squeeze()),
        'intercept': float(model.intercept_)
    }


def save_end_to_end_pipeline(pipeline, path):
    pipeline_dict = {
        'preprocessor': make_representation(pipeline.steps[0][1]),
        'predictor': make_representation(pipeline.steps[-1][1])
    }
    if len(pipeline.steps) > 2:
        pipeline_dict['commonTransformations'] = make_representation(Pipeline(pipeline.steps[1:-1]))
    with open(path, 'wt') as writing_file:
        json.dump(pipeline_dict, writing_file, indent='  ')


CLASS_REPRESENTATION_GENERATORS = {
    Pipeline: representation_transformers_pipeline,
    ColumnTransformer: representation_column_transformer,
    LogisticRegression: representation_logistic_regression,
    SimpleImputer: representation_imputer,
    OneHotEncoder: representation_one_hot,
    MinMaxScaler: representation_scaler
}


def make_representation(transformer):
    transformer_class = type(transformer)
    if transformer_class not in CLASS_REPRESENTATION_GENERATORS:
        raise ValueError(f'Unsupported transformer type: {str(transformer_class)}')
    representation_generator = CLASS_REPRESENTATION_GENERATORS[transformer_class]
    return representation_generator(transformer)
