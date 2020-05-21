import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#0dropout_rate
#1test_batchsize
#2k
#3lr
#4momentum
#5weight_decay



def loadData(path):
    df = pd.DataFrame()
    bestiteration = []
    for foldername in os.listdir(path):
            #Load main frame with hyperarams
            mainFrame = pd.read_csv(fr"{path}\{foldername}\hyperparameters.txt", delimiter=":", header=None,names=[0,foldername],dtype={f'{foldername}':np.float64})
            mainFrame.drop(columns=0,inplace=True)
            
            #Retrieve number of best epoch
            bestEpoch = pd.read_csv(fr"{path}\{foldername}\epoch.txt", header=None)
            bestEpoch = bestEpoch.values[0][0]
            mainFrame.loc['BestEpoch']=int(bestEpoch+1)

            #Retrieve acc for best epoch
            test_Accuracy = pd.read_csv(fr"{path}\{foldername}\Test_Accuracy_normal.csv", header=None)
            mainFrame.loc['Eval acc']= test_Accuracy.values[int(bestEpoch)][0]
            
            test_loss = pd.read_csv(fr"{path}\{foldername}\test_Loss.csv", header=None)
            mainFrame.loc['Eval loss']= test_loss.values[int(bestEpoch)][0]

            train_Accuracy = pd.read_csv(fr"{path}\{foldername}\Train_Accuracy_normal.csv", header=None)
            mainFrame.loc['Train acc']= train_Accuracy.values[int(bestEpoch)][0]
            
            train_loss = pd.read_csv(fr"{path}\{foldername}\train_Loss.csv", header=None)
            mainFrame.loc['Train loss']= train_loss.values[int(bestEpoch)][0]

            
            #test_acc,total_time,tn,fp,fn,tp

            RealTest = pd.read_csv(fr"{path}\{foldername}\RealTest.csv", header=None)
            RealTestAcc =  RealTest.values[0][0]
            RealTestTime = RealTest.values[1][0]
            tn,fp,fn,tp = RealTest.values[2][0],RealTest.values[3][0],RealTest.values[4][0],RealTest.values[5][0]
            mainFrame.loc['Test Acc']= RealTestAcc
            mainFrame.loc['Test Time']= RealTestTime
            mainFrame.loc['TN']= tn
            mainFrame.loc['FP']= fp
            mainFrame.loc['FN']= fn
            mainFrame.loc['TP']= tp
            

            df = pd.concat([df,mainFrame],axis=1)
            
            #print(f"{foldername} Best epoch: {bestEpoch} Best epoch from max: {test_Accuracy.idxmax().values[0]} acc: {test_Accuracy.values[int(bestEpoch)][0]}")
            bestiteration.append([test_Accuracy.values[int(bestEpoch)][0]])
            
    df.dropna(inplace=True)
    df.rename(index={0:'dropout_rate',1:'train_batchsize',2:'test_batchsize',3:'k',4:'Learning rate',5:'Momentum',6:'Weight_Decay'},inplace=True)
    df = df.transpose()
    return df

def plotValues(data,value1,value2,xlabel,ylabel):
    value1Data = data[value1].values
    value2Data = data[value2].values
    plt.scatter(value1Data,value2Data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# def plotValues(data,value,xlabel,ylabel):
#     valueData = data[value].values
#     xaxis = np.linspace(1,valueData.size,num=valueData.size)
#     print(xaxis.shape)
#     print(valueData.size)
#     plt.scatter(xaxis,valueData)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()

def makeFigure(foldername,path):

    Train_Accuracy_normal = pd.read_csv(fr"{path}\{foldername}\Test_Accuracy_normal.csv", header=None)
    Test_Accuracy_normal = pd.read_csv(fr"{path}\{foldername}\Train_Accuracy_normal.csv", header=None)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(Train_Accuracy_normal, label='Train')
    plt.plot(Test_Accuracy_normal, label='Test')
    plt.legend()
    plt.show()
    return

def pairplot(data):
    data = data.sort_values("Test Acc")
    data1 = data.drop(["TN","FP","FN","TP","train_batchsize","test_batchsize","BestEpoch","Eval acc","Train acc","Train loss", "Eval loss"], axis=1)
    sns.pairplot(data1)
    plt.show()
    return

def checkAmountOfEpochs():
    i = 0
    for foldername in os.listdir(path1):
        test_Accuracy = pd.read_csv(fr"{path1}\{foldername}\Test_Accuracy_normal.csv", header=None)
        epochs = test_Accuracy.shape
        i=i+1
        print(foldername,epochs)
    print(i)
    return

data = loadData(r"D:\Aalborg Universitet\VGIS8 - Dokumenter\Project\Hyperparameters iterations\Moaaz")
data.to_csv(r'C:\Users\Lynge\Documents\Projects\SemesterProject\SemesterProject_843\Evaluation\graph.txt')

#pairplot(data)

#data= data.sort_values("k")
#plotValues(data,"k","Test Time","k","Test Time [s]")

#data= data.sort_values("Test Acc")
#plotValues(data,"Test Acc","Model","Test Accuracy")
#print(data.describe())
#'dropout_rate',1:'train_batchsize',2:'test_batchsize',3:'k',4:'Learning rate',5:'Momentum',6:'Weight_Decay'},inplace=True)
confusion = data[["Test Acc","TN","FP","FN","TP"]]
confusion = confusion.sort_values("Test Acc",ascending=False)
#confusion = confusion[["TN","FP","FN","TP"]]/4726*100

modeldata = data[["Train acc","Eval acc","Test Acc","TN","FP","FN","TP","k","Learning rate","dropout_rate",'Momentum','Weight_Decay']]
modeldata = modeldata.sort_values("Test Acc",ascending=False)

print(confusion)