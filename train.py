from feature_process import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def train(train_all=True):
    # TODO 1:读入dataset
    if train_all:
        dataset_x, dataset_y = get_dataset('./data/train.csv',read_all=True)
    else:
        dataset_x, dataset_y = get_dataset('./data/train.csv', read_all=False)

    # TODO 2:PCA降维
    pca = PCA(n_components='mle', svd_solver='full')
    dataset_x_pca = pca.fit_transform(dataset_x)
    joblib.dump(pca, './model/pca.m')

    # TODO 3:分割数据集为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(dataset_x_pca, dataset_y, test_size=0.3, random_state=0)

    # TODO 4:SVM Classifier
    svc = SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True)
    svc.fit(train_x, train_y)
    print(svc.score(test_x, test_y))

    # TODO 5:save model
    joblib.dump(svc, './model/train_model.m')

