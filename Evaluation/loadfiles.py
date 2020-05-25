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
    accFrame = pd.DataFrame()
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
            
            bestiteration.append([test_Accuracy.values[int(bestEpoch)][0]])
            
            #Accuracy Frame
            test_Accuracy = test_Accuracy.transpose()
            accFrame = accFrame.append(test_Accuracy, ignore_index=True)
            df = pd.concat([df,mainFrame],axis=1)


            
            #print(f"{foldername} Best epoch: {bestEpoch} Best epoch from max: {test_Accuracy.idxmax().values[0]} acc: {test_Accuracy.values[int(bestEpoch)][0]}")
            
            
            
    df.dropna(inplace=True)
    df.rename(index={0:'dropout_rate',1:'train_batchsize',2:'test_batchsize',3:'k',4:'Learning rate',5:'Momentum',6:'Weight_Decay'},inplace=True)
    df = df.transpose()
    return df, accFrame

def plotValues(data,value1,value2,value3,xlabel,ylabel):
    fig, ax = plt.subplots()
    value1Data = data[value1].values
    value2Data = data[value2].values
    value3Data = data[value3].values
    value2Data = value2Data*100
    value3Data = value3Data*100
    plt.scatter(value1Data,value2Data, label="Test", marker="x")
    plt.scatter(value1Data,value3Data, label="Validation",facecolors='none',edgecolors='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ymajor_ticks = np.arange(45, 100, 5)
    yminor_ticks = np.arange(45, 100, 1)
    xmajor_ticks = np.arange(15, 26, 1)
    
    ax.set_yticks(ymajor_ticks)
    ax.set_yticks(yminor_ticks,minor=True)
    ax.set_xticks(xmajor_ticks)

    #ax.set_yticks(yminor_ticks, minor=True)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy [%]")


 

    # Or if you want different settings for the grids:
    #ax.grid(which='minor', alpha=0.2)
    ax.grid(True,which='major', alpha=0.5)
    ax.grid(True,which='minor', alpha=0.2)
    ax.set_ylim(45,100)
    plt.show()

def makeFigure(foldername,path,title):

    Test_Accuracy_normal = pd.read_csv(fr"{path}\{foldername}\Test_Accuracy_normal.csv", header=None)
    Train_Accuracy_normal = pd.read_csv(fr"{path}\{foldername}\Train_Accuracy_normal.csv", header=None)
    
    Test_Accuracy_normal = Test_Accuracy_normal*100
    Train_Accuracy_normal = Train_Accuracy_normal*100

    fig, ax = plt.subplots()
    plt.title(title)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy [%]')
    ax.set_xlim(0,50)
    ax.set_ylim(45,100)
    ax.plot(Train_Accuracy_normal, label='Train')
    ax.plot(Test_Accuracy_normal, label='Validation')
    
    
    xmajor_ticks = np.arange(0, 50, 5)
    xminor_ticks = np.arange(0, 50, 1)
    ymajor_ticks = np.arange(45, 100, 5)
    yminor_ticks = np.arange(45, 100, 1)


    ax.set_xticks(xmajor_ticks)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(ymajor_ticks)
    ax.set_yticks(yminor_ticks, minor=True)

 

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(True,which='major', alpha=0.5)
    
    
    #plt.grid(axis='both')
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
def allAcc(dataAcc):
    dataAcc = dataAcc.transpose()
    dataAcc = dataAcc*100

    #ValidationAccAll
    xaxis = np.linspace(1,50,num=50)

    fig, ax = plt.subplots()
    plt.title("Validation accuracy for all models")
    ax.plot(xaxis,dataAcc)
    ax.set_xlim(0,50)
    ax.set_ylim(45,100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy [%]")

    xmajor_ticks = np.arange(0, 50, 5)
    xminor_ticks = np.arange(0, 50, 1)
    ymajor_ticks = np.arange(45, 100, 5)
    yminor_ticks = np.arange(45, 100, 1)


    ax.set_xticks(xmajor_ticks)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(ymajor_ticks)
    ax.set_yticks(yminor_ticks, minor=True)



    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(True,which='major', alpha=0.5)
    plt.show()

def boxplotting(data,value1,value2):
    value1Data = data[value1].values.astype(int)
    value2Data = data[value2].values

    #sns.set_style("whitegrid")
    print(value1Data)
    print(value2Data)
    ax = sns.boxplot(x=value1Data,y=value2Data)

    #xmajor_ticks = np.arange(15, 25, 1)
    #xminor_ticks = np.arange(15, 25, 1)
    ymajor_ticks = np.arange(25, 45, 1)
    #yminor_ticks = np.arange(25, 45, 0.5)


    #ax.set_xticks(xmajor_ticks)
    #ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(ymajor_ticks)
    #ax.set_yticks(yminor_ticks, minor=True)
    ax.set_xlabel("k")
    ax.set_ylabel("Time [s]")

 

    # Or if you want different settings for the grids:
    #ax.grid(which='minor', alpha=0.2)
    ax.grid(True,which='major', alpha=0.5)
    plt.show()
data, dataAcc = loadData(r"D:\Aalborg Universitet\VGIS8 - Dokumenter\Project\Hyperparameters iterations\Moaaz")
data.to_csv(r'C:\Users\Lynge\Documents\Projects\SemesterProject\SemesterProject_843\Evaluation\graph.txt')

#sns.clustermap(data)

#boxplotting(data,"k","Test Time")
plotValues(data,"k","Test Acc","Eval acc","K","Accuracy")

#plotValues(data,"k","Eval acc","Validation Accuracy","k")

confusion = data[["Test Acc","TN","FP","FN","TP"]]
confusion = confusion.sort_values("Test Acc",ascending=False)

modeldata = data[["Train acc","Eval acc","Test Acc","TN","FP","FN","TP","k","Learning rate","dropout_rate",'Momentum','Weight_Decay']]
modeldata = modeldata.sort_values("Eval acc",ascending=False)

print(modeldata)