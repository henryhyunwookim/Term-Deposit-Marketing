import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler


def pivot_data(data, target, binary_target_value=None):
    X_original = data.drop([target], axis=1)
    y_original = data[target]
    y = y_original.apply(lambda x: 1 if x==binary_target_value else 0)

    X = pd.DataFrame()
    for col in X_original.columns:
        if type(X_original[col][0]) == str:
            col_pivoted = pd.get_dummies(X_original[col], prefix=col)
            X = pd.concat([X, col_pivoted], axis=1)
        else:
            X = pd.concat([X, X_original[col]], axis=1)

    return X, y


def resample_data(X_train, y_train, method, random_state=None):
    if method == "upsample":
        X_train_balanced, y_train_balanced = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    elif method == "downsample":
        X_train_balanced, y_train_balanced = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced


def normalize_data(X_train, X_test, method="standard"):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_test_normalized