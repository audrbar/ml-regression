# This is a main Python script.
import os

from data.datasets_data import random_dataset
from src.utils import plot_dataset, split_dataset, read_csv, choose_x_y

if __name__ == '__main__':
    path = os.getcwd()
    customers = read_csv(path + "/data/ecommerce_customers.csv")
    # Features (X), Target (y)
    X, y = choose_x_y(customers, 'Yearly Amount Spent', ['Avg. Session Length',
                                                         'Time on App', 'Time on Website',
                                                         'Length of Membership'])

    # X, y = random_dataset(200)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    plot_dataset(X['Time on App'], y)
