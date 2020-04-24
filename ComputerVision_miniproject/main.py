#This is for the main code

#Main(training): https://github.com/WangYueFt/dgcnn/blob/master/pytorch/main.py

#Data (Download dataset and get points): https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py

from model import DGCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from data import ModelNet40
import torch.nn.functional as F
from util import cal_loss
import sklearn.metrics as metrics
from tqdm import tqdm

NUM_CLASS = 40
EMB_DIMS = 1024
NUM_POINTS = 1024
DROPOUT_RATE = 0.5
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EPOCHS = 50
K = 20
LR = 0.001

device = torch.device("cuda")

model = DGCNN(numClass = NUM_CLASS, emb_dims = EMB_DIMS, dropout_rate=DROPOUT_RATE, batch_size = TRAIN_BATCH_SIZE, k = K).to(device)

def train():
    #Loading the dataloaders for train and test
    train_loader = DataLoader(ModelNet40(partition='train', num_points=NUM_POINTS), num_workers=8,
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=NUM_POINTS), num_workers=8,
                             batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)

    #Optimizer
    optimizer = optim.SGD(model.parameters(),lr = LR*100 , momentum = 0.9, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min= LR)

    #Loss function
    criterion = cal_loss #FOR US: Maybe just overwrite this with a function instead of calling into utils, as we just use one line from the other script. 
    #criterion = F.cross_entropy

    #Lists to save information such as accuracy and loss for each epoch.
    best_test_accuracy = 0
    
    train_epoch_accuracy_normal = []
    train_epoch_accuracy_average = []
    train_epoch_loss = []

    test_epoch_accuracy_normal = []
    test_epoch_accuracy_average = []
    test_epoch_loss = []

    for epoch in range(EPOCHS):
        #Training starts here
        print('[INFO] Training...')        
        scheduler.step()

        train_loss = 0.0
        count = 0.0
        print("Training start ln45")
        model.train()
        train_pred = []
        train_true = []
        
        print("Cuda: ",torch.cuda.get_device_name())
        
        for data, label in tqdm(train_loader):
            data = data.to(device) 
            label = label.to(device).squeeze()

            data = data.permute(0,2,1)
            batch_size = data.size()[0]
            
            optimizer.zero_grad() #zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).

            output = model(data) #Calculates class probabilities for each point cloud. So if batchsize is 16 and num classes is 40, it returns [16,40].
            loss = criterion(output,label,False) 
            loss.backward() #computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            optimizer.step() #causes the optimizer to take a step based on the gradients of the parameters.

            predictions = output.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item()*batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(predictions.detach().cpu().numpy())

        train_pred = np.concatenate(train_pred)
        train_true = np.concatenate(train_true)

        #Save accuracy and loss to a list
        normal_accuracy = metrics.accuracy_score(train_true,train_pred)
        average_accuracy = metrics.balanced_accuracy_score(train_true,train_pred)
        loss_value = train_loss*1.0/count

        train_epoch_accuracy_normal.append(normal_accuracy)
        train_epoch_accuracy_average.append(average_accuracy)
        train_epoch_loss.append(loss_value)
        print(f'Train {epoch}| Loss: {loss_value}| Train acc: {normal_accuracy}| Train avg acc: {average_accuracy}')
        
        np.savetxt('Train_Accuracy_normal.csv',np.array(train_epoch_accuracy_normal))
        np.savetxt('Train_Accuracy_average.csv',np.array(train_epoch_accuracy_average))
        np.savetxt('Train_Loss.csv',np.array(train_epoch_loss))

        #Training stops here

        #Testing starts here
        test_loss = 0.0
        count = 0.0
        print("Testing starts ln45")
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader):
            data = data.to(device) 
            label = label.to(device).squeeze()

            data = data.permute(0,2,1)
            batch_size = data.size()[0]            

            output = model(data) #Calculates class probabilities for each point cloud. So if batchsize is 16 and num classes is 40, it returns [16,40].
            loss = criterion(output,label,False)     

            predictions = output.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item()*batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(predictions.detach().cpu().numpy())
        test_pred = np.concatenate(test_pred)
        test_true = np.concatenate(test_true)

        #Save accuracy and loss to a list
        normal_accuracy = metrics.accuracy_score(test_true,test_pred)
        average_accuracy = metrics.balanced_accuracy_score(test_true,test_pred)
        loss_value = test_loss*1.0/count

        test_epoch_accuracy_normal.append(normal_accuracy)
        test_epoch_accuracy_average.append(average_accuracy)
        test_epoch_loss.append(loss_value)
        print(f'Test {epoch}| Loss: {loss_value}| Test acc: {normal_accuracy}| Test avg acc: {average_accuracy}')
        
        np.savetxt('Test_Accuracy_normal.csv',np.array(test_epoch_accuracy_normal))
        np.savetxt('Test_Accuracy_average.csv',np.array(test_epoch_accuracy_average))
        np.savetxt('Test_Loss.csv',np.array(test_epoch_loss))

        #Testing stops here

        #The model with the highest found accuracy for testing is saved
        if normal_accuracy > best_test_accuracy:
            best_test_accuracy = normal_accuracy 
            print('[INFO] Saving Model...')
            np.savetxt('Epoch.txt',np.array([epoch]))
            torch.save(model.state_dict(),'model.t7')

        


def test():
    print('[INFO] Testing...')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=NUM_POINTS), num_workers=8, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)

    test_model = DGCNN(numClass = NUM_CLASS, emb_dims = EMB_DIMS, dropout_rate=DROPOUT_RATE, batch_size = TRAIN_BATCH_SIZE, k = K).to(device)
    test_model.load_state_dict(torch.load('model.t7'))
    test_model.eval()

    criterion = cal_loss

    #Test
    test_loss = 0.0
    count = 0.0
    test_pred = []
    test_true = []
    for data, label in tqdm(test_loader):
        data = data.to(device) 
        label = label.to(device).squeeze()

        data = data.permute(0,2,1)
        batch_size = data.size()[0]            

        output = model(data) #Calculates class probabilities for each point cloud. So if batchsize is 16 and num classes is 40, it returns [16,40].
        loss = criterion(output,label,False)     

        predictions = output.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item()*batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(predictions.detach().cpu().numpy())
    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)

    print(f'Testing | Loss: {test_loss*1.0/count}| Test acc: {metrics.accuracy_score(test_true,test_pred)}| Test avg acc: {metrics.balanced_accuracy_score(test_true,test_pred)}')



# def train1(model, mode, dataloader, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
#     print('[INFO] {mode}...')
#     #Selecting which mode to run the model (Training or Testing)
#     if mode == 'Training':        
#         scheduler.step()
#         model.train()
#     else:
#         model.eval()

#     loss_sum = 0.0
#     count = 0.0
#     predicted_labels = []
#     correct_labels = []
    
#     for data, label in tqdm(dataloader):
#         data = data.to(device) 
#         label = label.to(device).squeeze()

#         data = data.permute(0,2,1)
#         batch_size = data.size()[0]
        
#         if mode == 'Training':
#             optimizer.zero_grad() #zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).

#         output = model(data) #Calculates class probabilities for each point cloud. So if batchsize is 16 and num classes is 40, it returns [16,40].
#         loss = criterion(output,label,False) 

#         if mode == 'Training':
#             loss.backward() #computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
#             optimizer.step() #causes the optimizer to take a step based on the gradients of the parameters.

#         predictions = output.max(dim=1)[1]
#         count += batch_size

#         loss_sum += loss.item()*batch_size
#         correct_labels.append(label.cpu().numpy())
#         predicted_labels.append(predictions.detach().cpu().numpy()) 

#     #Concats 
#     predicted_labels = np.concatenate(predicted_labels)
#     correct_labels = np.concatenate(correct_labels)

#     #Computes accuracy and loss based on metrics
#     normal_accuracy = metrics.accuracy_score(correct_labels,predicted_labels)
#     average_accuracy = metrics.balanced_accuracy_score(correct_labels,predicted_labels)
#     loss_sum = loss_sum*1.0/count

#     #Save accuracy and loss to lists
#     epoch_accuracy_normal.append(normal_accuracy)
#     epoch_accuracy_average.append(average_accuracy)
#     epoch_loss.append(loss_sum)
    
#     print(f'{mode} {epoch}| Loss: {loss_sum}| {mode} acc: {normal_accuracy}| {mode} avg acc: {average_accuracy}')

#     #The model with the highest found accuracy for testing is saved
#     if mode == 'Testing':
#         if normal_accuracy > best_test_accuracy:
#             best_test_accuracy = normal_accuracy 
#             print('[INFO] Saving Model...')
#             np.savetxt('Epoch.txt',np.array([epoch]))
#             torch.save(model.state_dict(),'model.t7')
#             return best_test_accuracy


# def save_to_csvs(mode, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
#     np.savetxt(f'{mode}_Accuracy_normal.csv',np.array(epoch_accuracy_normal))
#     np.savetxt(f'{mode}_Accuracy_average.csv',np.array(epoch_accuracy_average))
#     np.savetxt(f'{mode}_Loss.csv',np.array(epoch_loss))


# def new_main():
#     model = DGCNN(numClass = NUM_CLASS, emb_dims = EMB_DIMS, dropout_rate=DROPOUT_RATE, batch_size = TRAIN_BATCH_SIZE, k = K).to(device)

#     device = torch.device("cuda")
#     print("Cuda: ",torch.cuda.get_device_name())

#     #Loading the dataloaders for train and test
#     train_loader = DataLoader(ModelNet40(partition='train', num_points=NUM_POINTS), num_workers=8,
#                               batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
#     test_loader = DataLoader(ModelNet40(partition='test', num_points=NUM_POINTS), num_workers=8,
#                              batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)

#     #Optimizer
#     optimizer = optim.SGD(model.parameters(),lr = LR*100 , momentum = 0.9, weight_decay = 1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min= LR)

#     #Loss function
#     criterion = cal_loss #FOR US: Maybe just overwrite this with a function instead of calling into utils, as we just use one line from the other script. 

#     #Lists to save information such as accuracy and loss for each epoch.
#     best_test_accuracy = 0
    
#     train_epoch_accuracy_normal = []
#     train_epoch_accuracy_average = []
#     train_epoch_loss = []

#     test_epoch_accuracy_normal = []
#     test_epoch_accuracy_average = []
#     test_epoch_loss = []

#     for epoch in range(EPOCHS):
#         train1(model, 'Training', train_loader, train_epoch_accuracy_normal, train_epoch_accuracy_average, train_epoch_loss)
#         best_test_accuracy = train1(model, 'Testing', test_loader, test_epoch_accuracy_normal, test_epoch_accuracy_average, test_epoch_loss)

#     #Saves the accuracy and lost lists into csvs
#     save_to_csvs(train_epoch_accuracy_normal, train_epoch_accuracy_average, train_epoch_loss)
#     save_to_csvs(test_epoch_accuracy_normal, test_epoch_accuracy_average, test_epoch_loss)

if __name__ == "__main__":
    #new_main():
    train()
    test()
    
    
    
