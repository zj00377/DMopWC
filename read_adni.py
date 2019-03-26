from scipy.io import loadmat
import numpy as np
from sklearn.utils import shuffle
np.random.seed(15)

def load_ADNI(label, data, train_rate, mode):
    y = loadmat('MMSE_ADAS_update.mat')
    Y = []
    if label == "MMSE":
        Y.append(y['MMSE_base'])
        Y.append(y['MMSE_m06'])
        Y.append(y['MMSE_m12'])
        Y.append(y['MMSE_m24'])
    elif label == "ADAS":
        Y.append(y['ADAS_base'])
        Y.append(y['ADAS_m06'])
        Y.append(y['ADAS_m12'])
        Y.append(y['ADAS_m24'])
    print("Finish Load {} label!".format(label))
    
    if data == "mTBM":
        x = loadmat('ADNI.mat')
        X = []
        X.append(x['base'])
        X.append(x['m06'])
        X.append(x['m12'])
        X.append(x['m24'])
        print("Finish load ADNI {} data!".format(data))
        trainXs, trainYs, testXs, testYs = [], [], [], []
        for i in range(4):
            train_num = int(X[i].shape[0] * train_rate)
            temp = np.reshape(X[i], (X[i].shape[0], 120000))
            trainXs.append(temp[0:train_num-1, :])
            trainYs.append(Y[i][0:train_num-1, :]) 
            testXs.append(temp[train_num:, :])
            testYs.append(Y[i][train_num:, :]) 
    elif data == "patch":
        x = np.loadtxt('/media/qdong17/Data/ASU_Research/MMDL/data/ADNI_base.txt')
        X = []
        X.append(np.transpose(x))
        print(X[0].shape)
        print("Finish load ADNI {} data!".format(data))
        trainXs, trainYs, testXs, testYs = [], [], [], []
        for i in range(1):
            train_num = int(X[i].shape[0] * train_rate)
            trainXs.append(X[i][0:train_num-1, :])
            trainYs.append(Y[i][0:train_num-1, :]) 
            testXs.append(X[i][train_num:, :])
            testYs.append(Y[i][train_num:, :]) 

    elif data == "sparse":
        x = np.loadtxt('4Stage1.txt')
        X = []
        X.append(np.transpose(x))
        x = np.loadtxt('4Stage2.txt')
        X.append(np.transpose(x))
        x = np.loadtxt('4Stage3.txt')
        X.append(np.transpose(x))
        x = np.loadtxt('4Stage4.txt')
        X.append(np.transpose(x))
        print("Finish load ADNI {} data!".format(data))
        trainXs, trainYs, testXs, testYs = [], [], [], []
        for i in range(4):
            randomize = np.arange(len(X[i]))
            np.random.shuffle(randomize)
            tempX, tempY = X[i][randomize], Y[i][randomize]
            train_num = int(X[i].shape[0] * train_rate)
            trainXs.append(X[i][0:train_num-1, :])
            trainYs.append(Y[i][0:train_num-1, :]) 
            testXs.append(X[i][train_num:, :])
            testYs.append(Y[i][train_num:, :]) 

    l = np.where(trainYs[0][:, 0]==0)
    l = np.array(l)
    trainXs[0] = np.delete(trainXs[0], l, axis=0)
    trainYs[0] = np.delete(trainYs[0], l, axis=0)
    l = np.where(testYs[0][:, 0]==0)
    l = np.array(l)
    testXs[0] = np.delete(testXs[0], l, axis=0)
    testYs[0] = np.delete(testYs[0], l, axis=0)

    print("Finish create 4 tasks!")
    return trainXs, trainYs, testXs, testYs
                          
        




