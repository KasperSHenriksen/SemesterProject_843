#This is for the main code

#Main(training): https://github.com/WangYueFt/dgcnn/blob/master/pytorch/main.py

#Data (Download dataset and get points): https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py

from model import DGCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

NUM_CLASS = 40
EMB_DIMS = 1024
NUM_POINTS = 1024
DROPOUT_RATE = 0.5
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
EPOCHS = 100
K = 20
LR = 0.001

device = torch.device("cuda")

model = DGCNN(numClass = NUM_CLASS, emb_dims = EMB_DIMS, dropout_rate=DROPOUT_RATE, batch_size = TRAIN_BATCH_SIZE, k = K).to(device)

def train():
    train_loader = DataLoader(ModelNet40(partition='train', num_points=NUM_POINTS), num_workers=8,
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=NUM_POINTS), num_workers=8,
                             batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)

    optimizer = optim.SGD(model.parameters(),lr = LR*100 , momentum = 0.9, weight_decay = 1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min= LR)

    criterion = nn.CrossEntropyLoss

    for epoch in range(EPOCHS):
        scheduler.step()        

        train_loss = 0.0
        count = 0.0

        model.train()
        train_pred = []
        train_true = []

        for data, label in train_loader():
            data, label = data.to(device), label.to(device).squeeze()
            
            print(f'Before: {data}')

            data = data.permute(0,2,1)
            
            print(f'After: {data')

            batch_size = data.size()[]