import pandas as pd
import os 
import matplotlib.pyplot as plt

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
            mainFrame = pd.read_csv(fr"{path}\{foldername}\hyperparameters.txt", delimiter=":", header=None,names=[0,foldername])
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
            
            mainFrame.loc['Test Acc']= RealTestAcc
            mainFrame.loc['Test Time']= RealTestTime
            mainFrame.loc['TN']= 0
            mainFrame.loc['FP']= 0
            mainFrame.loc['FN']= 0
            mainFrame.loc['TP']= 0
            

            df = pd.concat([df,mainFrame],axis=1)
            
            #print(f"{foldername} Best epoch: {bestEpoch} Best epoch from max: {test_Accuracy.idxmax().values[0]} acc: {test_Accuracy.values[int(bestEpoch)][0]}")
            bestiteration.append([test_Accuracy.values[int(bestEpoch)][0]])
            
    df.dropna(inplace=True)
    df.rename(index={0:'dropout_rate',1:'train_batchsize',2:'test_batchsize',3:'k',4:'Learning rate',5:'Momentum',6:'Weight_Decay'},inplace=True)
    return df



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

data = loadData(r"D:\Aalborg Universitet\VGIS8 - Dokumenter\Project\Hyperparameters iterations\Moaaz")
data.to_csv(r'C:\Users\Lynge\Documents\Projects\SemesterProject\SemesterProject_843\Evaluation\graph.csv')

temp = data.iloc[8]
#makeFigure('Iteration (2)',r"D:\Aalborg Universitet\VGIS8 - Dokumenter\Project\Hyperparameters iterations\Moaaz")
print(data)
print(temp.values)
#print(temp)
plt.plot.set_marker(".")
plt.plot(temp.values)

plt.show()



#Tjek how many epoch in each iteration
def checkAmountOfEpochs():
    i = 0
    for foldername in os.listdir(path1):
        test_Accuracy = pd.read_csv(fr"{path1}\{foldername}\Test_Accuracy_normal.csv", header=None)
        epochs = test_Accuracy.shape
        i=i+1
        print(foldername,epochs)
    print(i)



