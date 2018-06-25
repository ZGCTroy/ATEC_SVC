from feature_process import *
from sklearn.externals import joblib
from sklearn.decomposition import PCA


def predict(predict_all=True):
    # TODO 1:get dataset
    if predict_all:
        id, predict_x = get_predict_dataset('./data/atec_anti_fraud_test_a.csv', read_all=True)
    else:
        id, predict_x = get_predict_dataset('./data/atec_anti_fraud_test_a.csv', read_all=False)

    # TODO 2: PCA
    pca = joblib.load('./model/pca.m')
    predict_x = pca.transform(predict_x)

    # TODO 3:load a pre-trained model
    model = joblib.load('./model/train_model.m')

    # TODO 4:predict
    predictions = model.predict_proba(predict_x)[:, 1]

    # TODO 7: save result as .csv
    result = pd.DataFrame({'id': id, 'score': predictions})
    result.to_csv('./result/svc.csv', index=False)
