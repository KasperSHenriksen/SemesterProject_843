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

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(Train_Accuracy_normal, label='Train')
plt.plot(Test_Accuracy_normal, label='Test')
plt.plot(Train_Accuracy_average, label='Train avg')
plt.plot(Test_Accuracy_average, label='Test avg')
plt.legend()
#plt.show()
plt.savefig('accuracy.png')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(Train_Loss, label='Train Loss')
plt.plot(Test_Loss, label='Test Loss')
plt.legend()
#plt.show()
plt.savefig('loss.png')