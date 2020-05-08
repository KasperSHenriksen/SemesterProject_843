import pandas as pd
import matplotlib.pyplot as plt
Test_Accuracy_average = pd.read_csv('Test_Accuracy_average.csv',header=None)
Test_Accuracy_normal = pd.read_csv('Test_Accuracy_normal.csv',header=None)
Test_Loss = pd.read_csv('Test_Loss.csv',header=None)
Train_Accuracy_average = pd.read_csv('Train_Accuracy_average.csv',header=None)
Train_Accuracy_normal = pd.read_csv('Train_Accuracy_normal.csv',header=None)
Train_Loss = pd.read_csv('Train_Loss.csv',header=None)

epoch = 1
print('Test acc: ',float(Test_Accuracy_normal.loc[epoch]))
print('Test acc avg',float(Test_Accuracy_average.loc[epoch]))
print('Train acc: ',float(Train_Accuracy_normal.loc[epoch]))
print('Train acc avg: ',float(Train_Accuracy_average.loc[epoch]))
print('Train Loss: ',float(Train_Loss.loc[epoch]))
print('Test Loss: ',float(Test_Loss.loc[epoch]))


fig_accuracy = plt.figure()
fig_accuracy.xlabel('Epochs')
fig_accuracy.ylabel('Accuracy')
fig_accuracy.plot(Train_Accuracy_normal, label='Train')
fig_accuracy.plot(Test_Accuracy_normal, label='Test')
fig_accuracy.plot(Train_Accuracy_average, label='Train avg')
fig_accuracy.plot(Test_Accuracy_average, label='Test avg')
fig_accuracy.legend()
#plt.show()
fig_accuracy.savefig('accuracy.png')

fig_loss = plt.figure()
fig_loss.xlabel('Epochs')
fig_loss.ylabel('Loss')
fig_loss.plot(Train_Loss, label='Train Loss')
fig_loss.plot(Test_Loss, label='Test Loss')
fig_loss.legend()
#plt.show()
fig_loss.savefig('loss.png')