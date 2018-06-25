from feature_process import *
from predict import *
from train import *


def main():
    # TODO 1: 模型训练
    train(train_all=False)

    # TODO 2: 预测并输出结果至./result/svc.csv
    predict(predict_all=False)


if __name__ == '__main__':
    main()
