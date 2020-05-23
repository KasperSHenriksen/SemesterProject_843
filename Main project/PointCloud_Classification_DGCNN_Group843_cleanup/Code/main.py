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
import time
from sklearn.metrics import confusion_matrix as sklean_confusion_matrix

#Static Parameters
NUM_CLASS = 2
EMB_DIMS = 1024
NUM_POINTS = 1024
EPOCHS = 50

def random_hyperparamter_search(current_path, dropout, train_batchsize, valid_batchsize, k, lr, momentum, weight_decay):
    dropout = np.random.uniform(dropout[0],dropout[1])
    train_batchsize = np.random.randint(train_batchsize[0],train_batchsize[1])
    valid_batchsize = np.random.randint(valid_batchsize[0],valid_batchsize[1])
    k = np.random.randint(k[0],k[1])

    lr_list = list(np.logspace(np.log10(lr[0]), np.log10(lr[1]), base = 10, num = 1000))
    lr = lr_list[np.random.randint(0,len(lr_list))]

    momentum = np.random.uniform(momentum[0],momentum[1])
    weight_decay = np.random.uniform(weight_decay[0],weight_decay[1])

    hyperparameters = {"dropout_rate" : dropout,
                        "train_batchsize" : train_batchsize,
                        "test_batchsize": valid_batchsize,
                        "k":k,
                        "lr":lr,
                        "momentum":momentum,
                        "weight_decay":weight_decay}

    with open(f'{current_path}/hyperparameters.txt', 'w') as f:
        for key in hyperparameters:
            f.write(f"{key}: {hyperparameters[key]}\n")

    return hyperparameters


def cal_loss(pred, label):
    label = label.contiguous().view(-1)
    loss = F.cross_entropy(pred, label, reduction='mean')
    return loss

def calculate_results(correct_labels, predicted_labels,loss_sum, results, mode):
    #Concats
    predicted_labels = np.concatenate(predicted_labels)
    correct_labels = np.concatenate(correct_labels)

    #Computes accuracy and loss based on metrics
    normal_accuracy = metrics.accuracy_score(correct_labels,predicted_labels)
    average_accuracy = metrics.balanced_accuracy_score(correct_labels,predicted_labels)

    #Save accuracy and loss to lists
    results.get('normal').append(normal_accuracy)
    results.get('average').append(average_accuracy)
    results.get('loss').append(loss_sum)

    print(f'{mode} {epoch}| Loss: {loss_sum}| {mode} acc: {normal_accuracy}| {mode} avg acc: {average_accuracy}')
    return normal_accuracy

def save_to_csvs(current_path, mode, results):
    np.savetxt(f'{current_path}/{mode}_Accuracy_normal.csv',np.array(results.get('normal')))
    np.savetxt(f'{current_path}/{mode}_Accuracy_average.csv',np.array(results.get('average')))
    np.savetxt(f'{current_path}/{mode}_Loss.csv',np.array(results.get('loss')))

def create_folder_and_get_path():
    num_subfolders_in_iterations = len(os.listdir("Iterations"))+1
    os.mkdir(f"Iterations/Iteration_{num_subfolders_in_iterations}")
    current_path = f"Iterations/Iteration_{num_subfolders_in_iterations}"
    return current_path

def save_highest_validation(normal_accuracy, best_valid_accuracy):
    if normal_accuracy > best_valid_accuracy:
        best_valid_accuracy = normal_accuracy
        print('[INFO] Saving Model...')
        np.savetxt(f'{current_path}/Epoch.txt', [epoch])
        torch.save(model,f'{current_path}/model.t7')
    return best_valid_accuracy



#Training
def train(current_path, model, epoch, mode, dataloader, optimizer, scheduler, criterion, best_valid_accuracy, results):
    print(f'[INFO] Epoch: {epoch}/{EPOCHS} | {mode}...')
    #Selecting which mode to run the model (Training or Testing)
    if mode == 'Training':
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
            scheduler.step() #Added here during clean up

        predictions = output.max(dim=1)[1] #This one
        count += batch_size

        loss_sum += loss.item()*batch_size
        correct_labels.append(label.cpu().numpy())
        predicted_labels.append(predictions.detach().cpu().numpy())

    loss_sum = loss_sum*1.0/count
    normal_accuracy = calculate_results(correct_labels, predicted_labels,loss_sum, results, mode)

    #The model with the highest found accuracy for testing is saved
    if mode == 'Validation':
        best_valid_accuracy = save_highest_validation(normal_accuracy, best_valid_accuracy)
        return best_valid_accuracy


#Testing
def test():
    TEST_BATCHSIZE = 16
    #NUM_POINTS = 1024

    test_loader = DataLoader(PointCloudDataset(partition='Testing', num_points=NUM_POINTS), batch_size=TEST_BATCHSIZE, shuffle=False, drop_last=False)

    path = r"E:\OneDrive\Aalborg Universitet\VGIS8 - Documents\Project\Hyperparameters iterations\Moaaz"
    interationFolders = os.listdir(path)
    for interationFolder in interationFolders:

        print(interationFolder)
        modelPath = fr"{path}\{interationFolder}\model.t7"
        # load model

        model = torch.load(modelPath)
        model.eval()

        start_time = time.time()
        predicted_labels = []
        correct_labels = []
        with torch.no_grad():
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
            #print(f'Correct: {correct_labels}')
            #print(f'Predicted: {predicted_labels}')
            test_acc = metrics.accuracy_score(correct_labels, predicted_labels)
            print(f'Test Accuracy = {test_acc}')

            #print("Confusion Matrix:",sklean_confusion_matrix(correct_labels,predicted_labels))
            tn, fp, fn, tp = sklean_confusion_matrix(correct_labels,predicted_labels).ravel()
            #print(tn, fp, fn, tp)

        stop_time = time.time()
        total_time = stop_time - start_time
        print(f"Time taken: {total_time}")
        np.savetxt(fr'{path}\{interationFolder}\RealTest.csv',np.array([test_acc,total_time,tn,fp,fn,tp]))

if __name__ == "__main__":
    #Select Cuda
    device = torch.device("cuda")
    print("Using CUDA with: ",torch.cuda.get_device_name())

    mode = input("Training or Testing: ")
    if mode == 'Training':
        current_path = create_folder_and_get_path() #Create new iteration folder to save information and then return the path to that folder.

        rdm_hyperparam = random_hyperparamter_search(current_path, dropout = [0.3,0.8], #Normal 0.5
                                                                    train_batchsize = [16,17], #Normal 16: 16 only max exclusive
                                                                    valid_batchsize = [8,9], #Normal 8: 8 only, max exclusive
                                                                    k = [15,26], #Normal 20: 15 to 25
                                                                    lr = [0.000001,0.001], #Normal 0.000001:
                                                                    momentum = [0.9,0.99], #Normal: 0.9
                                                                    weight_decay = [1e-5,1e-4]) #Normal: 1e-4

        #The DGCNN Model is instantiated here
        model = DGCNN(numClass = NUM_CLASS,
                      emb_dims = EMB_DIMS,
                      dropout_rate = rdm_hyperparam.get("dropout_rate"),
                      batch_size = rdm_hyperparam.get("train_batchsize"),
                      k = rdm_hyperparam.get("k")).to(device)

        #Data loaders
        train_loader = DataLoader(PointCloudDataset(partition='Training', num_points=NUM_POINTS), num_workers=8, batch_size=rdm_hyperparam.get("train_batchsize"), shuffle=True, drop_last=True) #Why drop last?
        valid_loader = DataLoader(PointCloudDataset(partition='Validation', num_points=NUM_POINTS), num_workers=8, batch_size=rdm_hyperparam.get("train_batchsize"), shuffle=True, drop_last=False)

        #Optimizer
        optimizer = optim.SGD(model.parameters(),lr = rdm_hyperparam.get("lr")*100 ,momentum = rdm_hyperparam.get("momentum"), weight_decay = rdm_hyperparam.get("weight_decay")) #0.9 normal momentum, LR*100

        #LR Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min= rdm_hyperparam.get("lr"))

        #Loss function
        criterion = cal_loss

        #Dicts to save information such as accuracy and loss for each epoch.
        best_valid_accuracy = 0
        tain_results = {'normal':[],'average':[],'loss':[]} #Maybe delete average if we don't use it
        valid_results = {'normal':[],'average':[],'loss':[]} #Maybe delete average if we don't use it

        #Go through all epochs
        for epoch in range(EPOCHS):
            #Training
            train(current_path, model, epoch, 'Training', train_loader, optimizer, scheduler, criterion, best_valid_accuracy, tain_results)
            #Validation
            best_valid_accuracy = train(current_path, model, epoch, 'Validation', valid_loader, optimizer, scheduler, criterion, best_valid_accuracy, valid_results)

            #Saves the accuracy and lost lists into csvs save_to_csvs(mode, epoch_accuracy_normal, epoch_accuracy_average, epoch_loss):
            save_to_csvs(current_path, 'Train',tain_results)
            save_to_csvs(current_path, 'Test',valid_results)
    elif mode == 'Testing':
        test()
