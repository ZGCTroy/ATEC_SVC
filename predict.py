from preprocess.feature_process import *
from sklearn.externals import joblib


def predict():
    # TODO 1:get dataset
    id, predict_x = get_predict_dataset('./data/test_b.csv')

    # TODO 3:load a pre-trained model
    model = joblib.load('./model/svc_without_classweight_1000000.m')

    # TODO 4:predict
    predictions = model.predict_proba(predict_x)[:, 1]

    # TODO 7: save result as .csv
    result = pd.DataFrame({'id': id, 'score': predictions})
    result.to_csv('./result/svc_without_classweight_1000000.csv', index=False)


if __name__ == '__main__':
    predict()
