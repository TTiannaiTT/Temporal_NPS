# AUC evaluation
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def acc_cal(prediction,label):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    result = prediction.argmax(1)
    count=0
    for i in range(len(label)):
        if result[i]==label[i]:
            count+=1
    return count/len(label)

def evalAUC(model,device,test_feature,test_label):
    test_feature = torch.tensor(test_feature, dtype=torch.float32).to(device)
    # print(test_feature)
    torch.set_printoptions(threshold=np.inf)
    predict_score = model(test_feature)[:,1]
    predict_score = predict_score.cpu()
    # print(predict_score)
    # print(test_label)
    auc = roc_auc_score(y_true=np.array(test_label), y_score=predict_score.detach().numpy())
    return auc

def evalAUCtest(model,device,test_feature,test_label):
    test_feature = torch.tensor(test_feature, dtype=torch.float32).to(device)
    # print(test_feature)
    torch.set_printoptions(threshold=np.inf)
    predict_score = model(test_feature)[:,1]
    predict_score = predict_score.cpu()
    # print(predict_score)
    # print(test_label)
    y = predict_score.detach().numpy()
    auc = roc_auc_score(y_true=np.array(test_label), y_score=y)
    print(auc)
    testy = np.ones([y.shape[0],1])
    auc = roc_auc_score(y_true=np.array(test_label), y_score=testy)
    print(auc)
    return auc

def checkAUC(model,device,test_feature,test_label):
    test_feature = torch.tensor(test_feature, dtype=torch.float32).to(device)
    # print(test_feature)
    torch.set_printoptions(threshold=np.inf)
    predict_score = model(test_feature)[:,1]
    predict_score = predict_score.cpu()
    print(predict_score)
    print(test_label)
    auc = roc_auc_score(y_true=np.array(test_label), y_score=predict_score.detach().numpy())
    print('bestauc: ',auc)
    return auc

def auc4BCE(model,device,test_feature,test_label):
    test_feature = torch.tensor(test_feature, dtype=torch.float32).to(device)
    # print(test_feature)
    torch.set_printoptions(threshold=np.inf)
    predict_score = model(test_feature)
    predict_score = predict_score.cpu()
    # print(predict_score)
    # print(test_label)
    auc = roc_auc_score(y_true=np.array(test_label), y_score=predict_score.detach().numpy())
    return auc

# y_true = [1,0,0]
# y_score = [0.5,0.5,0.5]
# auc = roc_auc_score(y_true, y_score)
# print(auc)
