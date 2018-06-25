from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer


def get_dataset(file, read_all=True):
    # TODO 1: read csv
    if read_all:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv(file, nrows=500)

    # TODO 2: drop useless feature
    data.drop(['id', 'date'], axis=1, inplace=True)

    # TODO 3: get x and y
    y = data['label'].values
    data.drop(['label'], axis=1, inplace=True)
    x = data.as_matrix()

    # TODO 4: processing the missing value
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    x = imp.fit_transform(x)

    # TODO 5: normalization
    x = preprocessing.scale(x)

    return x, y


def get_predict_dataset(file, read_all=True):
    # TODO 1: read csv
    if read_all:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv(file, nrows=500)

    # TODO 2: drop useless feature
    id = data['id'].values
    data.drop(['id', 'date'], axis=1, inplace=True)

    # TODO 3: get x and y
    x = data.values

    # TODO 4: processing the missing value
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    x = imp.fit_transform(x)

    # TODO 5: normalization
    x = preprocessing.scale(x)
    return id, x


def feature_process():
    dataset_x, dataset_y = get_dataset('./data/train_beta.csv',read_all=False)
    print(dataset_x)
    print(dataset_y)
    print()

    predict_x,predict_y = get_predict_dataset('./data/train_beta.csv',read_all=False)
    print(predict_x)
    print(predict_y)

if __name__ == '__main__':
    feature_process()
