from model import DGCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from data import PointCloudDataset
import torch.nn.functional as F
import sklearn.metrics as metrics
from tqdm import tqdm

#Static Parameters
NUM_CLASS = 2
EMB_DIMS = 1024
NUM_POINTS = 1024
DROPOUT_RATE = 0.5
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
EPOCHS = 1
K = 20 #20
LR = 0.00000001

device = torch.device("cuda")

model = DGCNN(numClass = NUM_CLASS, emb_dims = EMB_DIMS, dropout_rate=DROPOUT_RATE, batch_size = TRAIN_BATCH_SIZE, k = K).to(device)

#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(count_parameters(model))


def cal_loss(pred, label):
    label = label.contiguous().view(-1) #This One
    loss = F.cross_entropy(pred, label, reduction='mean') #BCELoss 
    #loss = nn.BCELoss(pred, label, reduction='mean') #BCELoss 
    return loss

def train(model, epoch, mode, dataloader, optimizer, scheduler, criterion, best_test_accuracy, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
    print(f'[INFO] {mode}...')
    #Selecting which mode to run the model (Training or Testing)
    if mode == 'Training':
        scheduler.step()
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    count = 0.0
    predicted_labels = []
    correct_labels = []

    for data, label in tqdm(dataloader):
        data = data.to(device)
        label = label.to(device).squeeze() #This one

        data = data.permute(0,2,1)
        batch_size = data.size()[0]

        if mode == 'Training':
            optimizer.zero_grad() #zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).

        output = model(data) #Calculates class probabilities for each point cloud. So if batchsize is 16 and num classes is 40, it returns [16,40].
        loss = criterion(output,label)

        if mode == 'Training':
            loss.backward() #computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            optimizer.step() #causes the optimizer to take a step based on the gradients of the parameters.

        predictions = output.max(dim=1)[1] #This one
        count += batch_size

        loss_sum += loss.item()*batch_size
        correct_labels.append(label.cpu().numpy())
        predicted_labels.append(predictions.detach().cpu().numpy())

    #Concats
    predicted_labels = np.concatenate(predicted_labels)
    correct_labels = np.concatenate(correct_labels)

    #Computes accuracy and loss based on metrics
    normal_accuracy = metrics.accuracy_score(correct_labels,predicted_labels)
    average_accuracy = metrics.balanced_accuracy_score(correct_labels,predicted_labels)
    loss_sum = loss_sum*1.0/count

    #Save accuracy and loss to lists
    epoch_accuracy_normal.append(normal_accuracy)
    epoch_accuracy_average.append(average_accuracy)
    epoch_loss.append(loss_sum)

    print(f'{mode} {epoch}| Loss: {loss_sum}| {mode} acc: {normal_accuracy}| {mode} avg acc: {average_accuracy}')

    #The model with the highest found accuracy for testing is saved
    if mode == 'TestingXXX':
        if normal_accuracy > best_test_accuracy:
            best_test_accuracy = normal_accuracy
            print('[INFO] Saving Model...')
            np.savetxt('Epoch.txt',np.array([epoch]))
            torch.save(model.state_dict(),'model.t7')
            return best_test_accuracy

def test(model):
    test_loader = DataLoader(PointCloudDataset(partition='Testing', num_points=NUM_POINTS), batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)

    model = model.eval()

    predicted_labels = []
    correct_labels = []
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device).squeeze()

        data = data.permute(0,2,1)
        batch_size = data.size()[0]

        output = model(data)
        preds = output.max(dim=1)[1]
        correct_labels.append(label.cpu().numpy())
        predicted_labels.append(preds.detach().cpu().numpy())
    correct_labels = np.concatenate(correct_labels)
    predicted_labels = np.concatenate(predicted_labels)
    print(f'Correct: {correct_labels}')
    print(f'Predicted: {predicted_labels}')
    test_acc = metrics.accuracy_score(correct_labels, predicted_labels)
    print(f'Test Accuracy = {test_acc}')

def save_to_csvs(mode, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
    np.savetxt(f'{mode}_Accuracy_normal.csv',np.array(epoch_accuracy_normal))
    np.savetxt(f'{mode}_Accuracy_average.csv',np.array(epoch_accuracy_average))
    np.savetxt(f'{mode}_Loss.csv',np.array(epoch_loss))

if __name__ == "__main__":
    print("Cuda: ",torch.cuda.get_device_name())

    #Loading the dataloaders for train and test
    #train_loader = DataLoader(ModelNet40(partition='train', num_points=NUM_POINTS), num_workers=8,
    #                          batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=NUM_POINTS), num_workers=8,
    #                         batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)
    train_loader = DataLoader(PointCloudDataset(partition='Training', num_points=NUM_POINTS), num_workers=8,
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(partition='Validation', num_points=NUM_POINTS), num_workers=8,
                             batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)

    #Optimizer
    optimizer = optim.SGD(model.parameters(),lr = LR*100 , momentum = 0.8, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min= LR)
    

    #Loss function
    #criterion = cal_loss
    criterion = cal_loss

    #Lists to save information such as accuracy and loss for each epoch.
    best_test_accuracy = 0

    train_epoch_accuracy_normal = []
    train_epoch_accuracy_average = []
    train_epoch_loss = []

    test_epoch_accuracy_normal = []
    test_epoch_accuracy_average = []
    test_epoch_loss = []

    #Go through all epochs
    for epoch in range(0):
        train(model, epoch, 'Training', train_loader, optimizer, scheduler, criterion, best_test_accuracy,
                                                                                       train_epoch_accuracy_normal,
                                                                                       train_epoch_accuracy_average,
                                                                                       train_epoch_loss)

        best_test_accuracy = train(model, epoch, 'Testing', test_loader, optimizer, scheduler, criterion, best_test_accuracy,
                                                                                                          test_epoch_accuracy_normal,
                                                                                                          test_epoch_accuracy_average,
                                                                                                          test_epoch_loss)

    #Saves the accuracy and lost lists into csvs save_to_csvs(mode, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
    save_to_csvs('Train',train_epoch_accuracy_normal, train_epoch_accuracy_average, train_epoch_loss)
    save_to_csvs('Test',test_epoch_accuracy_normal, test_epoch_accuracy_average, test_epoch_loss)

    #Testing
    test(model)