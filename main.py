from predict import *
from train import *




def main():
    # # TODO 1: 预处理数据（缺失值处理、PCA降维)
    # train_feature_process(train_all=True)
    # predict_feature_process(predict_all=True)

    # TODO 1: 模型训练
    train()

    # TODO 2: 预测并输出结果至./result/svc.csv
    predict()


if __name__ == '__main__':
    main()
