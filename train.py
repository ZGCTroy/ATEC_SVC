from preprocess.feature_process import *
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve


def mayi_score(y_true, y_score):
    """ Evaluation metric
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    score = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.01)[0][0]]

    return score

# 193 225 239 169 197 204
def train():
    # TODO 1: Model
    svc = SVC(
        #class_weight={0: 0.5035246727089627, 1: 75.42857142857143},
        probability=True,
        C=1
    )
    sgd = SGDClassifier(
        loss='log',
        penalty='l2',
        class_weight={0: 0.5035246727089627, 1: 71.42857142857143},
    )
    gdbt = GradientBoostingClassifier(n_estimators=200)

    # TODO 2:Test_Dataset
    reader = pd.read_csv('./data/train_without_sample_labeled.csv', iterator=True)
    data = reader.get_chunk(size=200000)
    test_y = data['label'].values.astype(int)
    data.drop(['label'], axis=1, inplace=True)
    test_x = data.values

    # TODO 3:Hyper parameters
    batch_size = 1000000
    batch = 0
    max_score = 0
    model = svc

    # TODO 4:train
    while 1:
        try:
            print('batch = ', batch)
            batch += 1
            # get x,y
            data = reader.get_chunk(size=batch_size)
            y = data['label'].values.astype(int)
            data.drop(['label'], axis=1, inplace=True)
            x = data.values

            # train model
            model.fit(x, y)

            # test model
            test_y_prob = model.predict_proba(X=test_x)[:, 1]
            score = mayi_score(test_y, test_y_prob)
            print('score = ', score)

            # save model
            if score > max_score:
                max_score = score
                joblib.dump(model, './model/svc_without_classweight_1000000.m')

        except StopIteration:
            print("Iteration is stopped")
            break

if __name__ == '__main__':
    train()
