from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# TODO 8 : knn
# print('start : knn')
# knn = KNeighborsClassifier(
#     n_neighbors=5,
#     algorithm='kd_tree',
#     leaf_size=30,
#     p=2,  # 欧氏距离
#     n_jobs=-1
# )
# knn.fit(
#     X=data.loc[data['label'] != -1, features_keeped].values,
#     y=data.loc[data['label'] != -1, 'label'].values
# )
# print(knn.predict(
#     X=data.loc[data['label'] == -1, features_keeped].values
# )
# )
# print('end : knn')

def feature_process(train_filepath, predict_filepath):
    # TODO 1 : read data and merge
    train_data = pd.read_csv(train_filepath)
    predict_data = pd.read_csv(predict_filepath)
    train_data.drop(['label', 'id'], axis=1, inplace=True)
    predict_data.drop(['id'],axis=1,inplace=True)
    data = pd.concat([train_data, predict_data])


    # TODO 2 : drop features that is missing more than 40%
    print('start : drop features that is missing more than 40% ')
    features_keeped = []
    features_droped = []
    for feature_name in data.columns:
        missing_ratio = np.sum(data[feature_name].isnull()) / len(data[feature_name])
        if missing_ratio < 0.6:
            features_keeped.append(feature_name)
        else:
            features_droped.append(feature_name)
    print('end : drop features : ', features_droped, '\n')

    # TODO 3 : fill the data
    print('start : fill the missing values')
    for feature_name in features_keeped:
        data[feature_name].fillna(data[feature_name].min(), inplace=True)
    print('end : filling succeed ! ', '\n')

    # TODO 4 : normalization
    print('start : normalization')
    ss = StandardScaler()
    ss.fit(data[features_keeped].values)
    print('end : normalization', '\n')

    return features_keeped, ss


def train_feature_process(filepath, features_keeped, ss):
    # TODO 1 : read data from a csv file
    data = pd.read_csv(filepath)

    # TODO 2 : drop the useless feature
    data = data[['label'] + features_keeped]

    # TODO 3 : fill the missing label
    length = len(data['label'])
    for i in range(0, length):
        if data['label'][i] == -1:
            data['label'][i] = 1

    # TODO 3 : sort by date in DESC
    print("start : sort data by 'date' in DESC")
    data = data.sort_values(by=['date'], ascending=False)
    print("end : sort successed" + '\n')

    # TODO 4 : fill the missing values and normalization
    print('start : fill the missing values')
    for feature_name in features_keeped:
        data[feature_name].fillna(data[feature_name].min(), inplace=True)
    print('end : filling succeed ! ', '\n')

    # TODO 5 : normalization
    print('start : normalization')
    data[features_keeped] = ss.transform(data[features_keeped].values)
    print('end : normalization', '\n')

    # TODO 9 : save to a new csv
    data.to_csv('./data/train_without_sample_labeled.csv', index=False)


def predict_feature_process(filepath, features_keeped, ss):
    # TODO 1 : read data from a csv file
    data = pd.read_csv(filepath)

    # TODO 2 : drop features that is missing more than 40%
    print('start : drop features that is missing more than 40% ')
    data = data[['id'] + features_keeped]
    print('end : drop features  ' '\n')

    # TODO 3 : fill the missing values
    print('start : fill the missing values')
    for feature_name in features_keeped:
        data[feature_name].fillna(data[feature_name].min(), inplace=True)
    print('end : filling succeed ! ', '\n')

    # TODO 4 : normalization
    print('start : normalization')
    data[features_keeped] = ss.transform(data[features_keeped].values)
    print('end : normalization', '\n')

    # TODO 7 : save to a new csv
    data.to_csv('./data/test_b.csv', index=False)


def get_train_dataset(filepath):
    # TODO 1 : read dataset from a csv
    data = pd.read_csv(filepath, nrows=500)

    # TODO 2 : split the data into x and y
    y = data['label'].values
    data.drop(['label'], axis=1, inplace=True)
    x = data.values

    return x, y


def get_predict_dataset(filepath):
    # TODO 1 : read dataset from a csv
    data = pd.read_csv(filepath)

    # TODO 2 : split the data into x and y
    id = data['id'].values
    data.drop(['id'], axis=1, inplace=True)
    x = data.values

    return id, x


if __name__ == '__main__':
    train_filepath = './data/origin_data/train_origin.csv'
    predict_filepath = './data/origin_data/test_b_origin.csv'

    features_keeped, ss = feature_process(train_filepath, predict_filepath)
    train_feature_process(train_filepath, features_keeped, ss)
    predict_feature_process(predict_filepath, features_keeped, ss)

    # train_x, train_y = get_train_dataset('./data/train_without_sample_labeled.csv')
    # predict_id, predict_x = get_predict_dataset('./data/test_b.csv')
