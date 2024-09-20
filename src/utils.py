import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def read_csv(path_to_file: str):
    """Reads data from .csv file to Pandas DataFrame

    :param path_to_file: str
    :return:
    """
    pd.options.display.max_columns = None
    df = pd.read_csv(path_to_file)
    print(df.head())
    print(df.info())
    print(df.describe())
    return df


def choose_x_y(df: pd.DataFrame, column_name_y: str, columns_names_X: list[str]):
    """Splits DataFrame to X and y values (lists).

    :param df: Pandas DataFrame
    :param column_name_y:
    :param columns_names_X:
    :return:
    """
    X = df[columns_names_X]
    y = df[column_name_y]
    print(X.info())
    print(y.info())
    return X, y


def split_dataset(X: list[float], y: list[float]):
    """Splits Dataset

    :param X: list[float]
    :param y: list[float]
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    return X_train, X_test, y_train, y_test


def plot_dataset(X: list[float], y: list[float]):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="b", marker="o", s=30)
    return plt.show()
