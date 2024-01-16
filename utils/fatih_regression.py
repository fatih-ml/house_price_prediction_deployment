import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression,
    LassoCV,
    Lasso,
    Ridge,
    RidgeCV,
    ElasticNetCV,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split


def calculate_score(y_train, y_train_pred, y_test, y_pred, model_name, cv_scores=None):
    """
    this function is serving for two purposes
    1. helping me the main function
    2. individual score calculation and evaluation of models
    model_name(string)
    when calling individually, dont specify the cv_scores
    """

    model_name_short = model_name[:5]

    scores = {
        f"{model_name_short}_train": {
            "R2": r2_score(y_train, y_train_pred),
            "-mae": mean_absolute_error(y_train, y_train_pred),
            "-mse": mean_squared_error(y_train, y_train_pred),
            "-rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        },
        f"{model_name_short}_test": {
            "R2": r2_score(y_test, y_pred),
            "-mae": mean_absolute_error(y_test, y_pred),
            "-mse": mean_squared_error(y_test, y_pred),
            "-rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        },
    }

    # Try-except block for the cross-validation scores
    try:
        cv_scores_mean = cv_scores.mean()
        scores[f"{model_name_short}_CV"] = {
            "R2": cv_scores_mean[0],
            "-mae": cv_scores_mean[1],
            "-mse": cv_scores_mean[2],
            "-rmse": cv_scores_mean[3],
        }
    except Exception as e:
        print(f"Error calculating CV scores: {e}")

    return pd.DataFrame(scores)


def calculate_scores(pipelines, X_train, X_test, y_train, y_test):
    """
    This function is taking a dictionary of pipelines
    and calculate 4 metrics related to regression models on both train and test dataset
    including cross validation scores
    returns a dataframe of comparison

    pipelines(dict): dictionary of pipelines
    returns pd.DataFrame
    """

    # make an empty dictionary and a dataFrame to stack the results
    model_scores_dict = {}
    counter_concat = 0
    model_scores = pd.DataFrame()
    scoring_metrics = [
        "r2",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
    ]

    # iterate through our stored pipelines in pipeline dictionary
    for model_name, pipeline in pipelines.items():
        # fit the pipelines
        pipeline.fit(X_train, y_train)

        # make predictions
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

        cv_scores = pd.DataFrame(
            cross_validate(pipeline, X_train, y_train, scoring=scoring_metrics)
        ).iloc[:, 2:]

        # calculate the scores
        scores_pipeline = calculate_score(
            y_train, y_train_pred, y_test, y_pred, model_name, cv_scores
        )

        # combine in a dataframe for comparison
        model_scores = pd.concat([model_scores, scores_pipeline], axis=1)

    return model_scores


def create_pipelines(algorithms_scaled, algorithms_unscaled, X, y):
    """
    This function a located at the core if this notebook since i will create pipelines many times,
    It simply takes the dataset of X and y
    And list of algorithms
    Returns a dictionary which is stacked with different pipelines

    algorithms(list)
    X(pd.DataFrame)
    y(pd.Series)
    """
    algorithms = algorithms_scaled + algorithms_unscaled

    pipelines = {}

    # Train - test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # discriminate the categorical and numerical features
    numerical_features = X.select_dtypes(include=["number"]).columns.to_list()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.to_list()

    # Create transformers for numerical and categorical columns
    scaler = StandardScaler()
    ohe_encoder = OneHotEncoder(handle_unknown="ignore")
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    # create preprocessor for algorithms need scaling and onehotencoding
    # i have already encoded Ordinal Categorical Variables
    preprocessor1 = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), numerical_features),
            (
                "ohe_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # create a preprocessor for tree based algorithms, they dont need scaling
    # better to use Ordinal Encoder inplace of OneHotEncoder since they are tree based algorithms
    preprocessor2 = ColumnTransformer(
        transformers=[
            (
                "ordinal_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
            ("passthrough", "passthrough", numerical_features),
        ],
        remainder="passthrough",
    )

    for algorithm in algorithms:
        # extract algorithm name as a string
        algorithm_name = algorithm.__class__.__name__

        # make pipelines for algorithms need scaling and onehotencoding (defined in preprocessor)
        if algorithm in algorithms_scaled:
            pipeline = Pipeline(
                steps=[("preprocessor1", preprocessor1), (algorithm_name, algorithm)]
            )

        # make pipelines for tree based algorithms (no need for scaling and onehot encoding, but only ordinalencoding)
        else:
            pipeline = Pipeline(
                steps=[("preprocessor2", preprocessor2), (algorithm_name, algorithm)]
            )

        # REUSABLE models
        # fill the dictionary with algo names and their corresponding pipelines for using later
        pipelines[algorithm_name] = pipeline

    return pipelines
