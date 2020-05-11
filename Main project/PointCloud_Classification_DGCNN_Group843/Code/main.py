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
import os

#Static Parameters
NUM_CLASS = 2
EMB_DIMS = 1024
NUM_POINTS = 1024
EPOCHS = 2

#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(count_parameters(model))

def random_hyperparamter_search(current_path, dropout, train_batchsize, test_batchsize, k, lr, momentum, weight_decay):
    dropout = np.random.uniform(dropout[0],dropout[1])
    train_batchsize = np.random.randint(train_batchsize[0],train_batchsize[1])
    test_batchsize = np.random.randint(test_batchsize[0],test_batchsize[1])
    k = np.random.randint(k[0],k[1])

    lr_list = list(np.logspace(np.log10(lr[0]), np.log10(lr[1]), base = 10, num = 1000))
    lr = lr_list[np.random.randint(0,len(lr_list))]

    momentum = np.random.uniform(momentum[0],momentum[1])
    weight_decay = np.random.uniform(weight_decay[0],weight_decay[1])

    hyperparameters = {"dropout_rate" : dropout,
                        "train_batchsize" : train_batchsize,
                        "test_batchsize": test_batchsize,
                        "k":k,
                        "lr":lr,
                        "momentum":momentum,
                        "weight_decay":weight_decay}

    with open(f'{current_path}/hyperparameters.txt', 'w') as f:
        for key in hyperparameters:
            f.write(f"{key}: {hyperparameters[key]}\n")

    return dropout, train_batchsize, test_batchsize, k, lr, momentum, weight_decay


def cal_loss(pred, label):
    label = label.contiguous().view(-1) #This One
    loss = F.cross_entropy(pred, label, reduction='mean') #BCELoss 
    #nn.BCEWithLogitsLoss
    #loss = nn.BCEWithLogitsLoss(pred, label, reduction='mean') #BCELoss 
    return loss

def train(current_path, model, epoch, mode, dataloader, optimizer, scheduler, criterion, best_test_accuracy, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
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
    if mode == 'Validation':
        if normal_accuracy > best_test_accuracy:
            best_test_accuracy = normal_accuracy
            print('[INFO] Saving Model...')
            np.savetxt(f'{current_path}/Epoch.txt', [epoch])
            torch.save(model,f'{current_path}/model.t7')
            return best_test_accuracy
def test(model):
    test_loader = DataLoader(PointCloudDataset(partition='Testing', num_points=NUM_POINTS), batch_size=TEST_BATCHSIZE, shuffle=True, drop_last=False)

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

def save_to_csvs(current_path, mode, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
    np.savetxt(f'{current_path}/{mode}_Accuracy_normal.csv',np.array(epoch_accuracy_normal))
    np.savetxt(f'{current_path}/{mode}_Accuracy_average.csv',np.array(epoch_accuracy_average))
    np.savetxt(f'{current_path}/{mode}_Loss.csv',np.array(epoch_loss))

if __name__ == "__main__":
    num_subfolders_in_iterations = len(os.listdir("Iterations"))+1
    os.mkdir(f"Iterations/Iteration_{num_subfolders_in_iterations}")
    current_path = f"Iterations/Iteration_{num_subfolders_in_iterations}"

    device = torch.device("cuda")
    print("Cuda: ",torch.cuda.get_device_name())

    print("running main ln157")

    DROPOUT_RATE, TRAIN_BATCHSIZE, TEST_BATCHSIZE, K, LR = random_hyperparamter_search(current_path, dropout = [0.5,0.8], #Normal 0.5
                                                                            train_batchsize = [16,17], #Normal 16: 16 only max exclusive
                                                                            test_batchsize = [8,9], #Normal 8: 8 only, max exclusive
                                                                            k = [15,26], #Normal 20: 15 to 25
                                                                            lr = [0.000001,0.001], #Normal 0.000001: 
                                                                            momentum = [0.9,0.99], #Normal: 0.9
                                                                            weight_decay = [1e-4,1e-5]) #Normal: 1e-4

    model = DGCNN(numClass = NUM_CLASS, emb_dims = EMB_DIMS, dropout_rate=DROPOUT_RATE, batch_size = TRAIN_BATCHSIZE, k = K).to(device)

    train_loader = DataLoader(PointCloudDataset(partition='Training', num_points=NUM_POINTS), num_workers=8,
                            batch_size=TRAIN_BATCHSIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(partition='Validation', num_points=NUM_POINTS), num_workers=8,
                            batch_size=TRAIN_BATCHSIZE, shuffle=True, drop_last=False)

    #Optimizer
    optimizer = optim.SGD(model.parameters(),lr = LR*100 , momentum = 0.9, weight_decay = 1e-4) #0.9 normal momentum, LR*100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min= LR)
    
    #Loss function
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

    for epoch in range(EPOCHS):
        print("running train ln186")
        train(current_path, model, epoch, 'Training', train_loader, optimizer, scheduler, criterion, best_test_accuracy,
                                                                                    train_epoch_accuracy_normal,
                                                                                    train_epoch_accuracy_average,
                                                                                    train_epoch_loss)

        best_test_accuracy = train(current_path, model, epoch, 'Validation', test_loader, optimizer, scheduler, criterion, best_test_accuracy,
                                                                                                        test_epoch_accuracy_normal,
                                                                                                        test_epoch_accuracy_average,
                                                                                                        test_epoch_loss)

        #Saves the accuracy and lost lists into csvs save_to_csvs(mode, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
        save_to_csvs(current_path, 'Train',train_epoch_accuracy_normal, train_epoch_accuracy_average, train_epoch_loss)
        save_to_csvs(current_path, 'Test',test_epoch_accuracy_normal, test_epoch_accuracy_average, test_epoch_loss)

    #Testing
    test(model)