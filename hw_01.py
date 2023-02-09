import sys

import numpy as np
import pandas as pd


def main():
    print("hello world")
    # load in csv file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    data = pd.read_csv(url, header=None)

    print(data)

    # average all classes
    average_sepal_len = np.average(data[0])
    print(average_sepal_len)

    # min all classes
    min_sepal_len = np.min(data[0])
    print(min_sepal_len)

    # max all classes
    max_sepal_len = np.max(data[0])
    print(max_sepal_len)

    # first-quartile all classes
    fir_q_sepal_len = np.quantile(data[0], 0.25)
    print(fir_q_sepal_len)

    # third-quartile all classes
    third_q_sepal_len = np.quantile(data[0], 0.75)
    print(third_q_sepal_len)


if __name__ == "__main__":
    sys.exit(main())
