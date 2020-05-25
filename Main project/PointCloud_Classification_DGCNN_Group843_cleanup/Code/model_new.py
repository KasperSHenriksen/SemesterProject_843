#Archiecture of DGCNN: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(data, k):
    inner = -2*torch.matmul(data.transpose(2,1),data)
    xx = torch.sum(data**2,dim=1,keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2,1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(data, k=20, idx=None):

    batch_size = data.size(0)
    num_points = data.size(2)

    #Selecting part of the data
    data = data.view(batch_size, -1, num_points)

    if idx is None:
        #Run knn to specify indexing
        idx = knn(data, k=k)
    device = torch.device('cuda')

    #Arange returns a 1-D tensor of size: batch_size,
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = data.size()

    data = data.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = data.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    data = data.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-data, data), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class EdgeConv(nn.Module):
    def __init__(self, k, channels_in, channels_out):
        super().__init__()

        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(channels_in,channels_out,kernel_size=1,bias=False),
                                    nn.BatchNorm2d(channels_out),
                                    nn.LeakyReLU(negative_slope=0.2))

    def forward(self, data):
        data = get_graph_feature(data, k=self.k)
        data = self.conv1(data)
        return data.max(dim=-1, keepdim=False)[0]


class DGCNN(nn.Module):
    def __init__(self, numClass, emb_dims, dropout_rate, batch_size, k):
        super(DGCNN,self).__init__() #Runs the init of the base class, which enables functions to be utilized from this class.

        #Arguments:
        self.emb_dims = emb_dims
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.k = k

        #Conv Layers
        self.edge_conv1 = EdgeConv(k, channels_in = 6, channels_out = 64)
        self.edge_conv2 = EdgeConv(k, channels_in = 64*2, channels_out = 64)
        self.edge_conv3 = EdgeConv(k, channels_in = 64*2, channels_out = 128)
        self.edge_conv4 = EdgeConv(k, channels_in = 128*2, channels_out = 256)

        self.conv1 = nn.Sequential(nn.Conv1d(512,self.emb_dims,kernel_size=1,bias=False), nn.BatchNorm1d(self.emb_dims), nn.LeakyReLU(negative_slope=0.2))

        #Fully Connected layers| Linear == Fully connected
        self.fc1 = nn.Sequential(nn.Linear(self.emb_dims*2,512,bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2))
        self.fc2 = nn.Sequential(nn.Linear(512,256), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.fc3 = nn.Linear(256,numClass)

        #Dropout Layers
        self.dp1 = nn.Dropout(self.dropout_rate)
        self.dp2 = nn.Dropout(self.dropout_rate)



    def forward(self, data):
        batch_size = data.size(0)

        #EdgeConv1
        data1 = self.edge_conv1(data)

        #EdgeConv2
        data2= self.edge_conv2(data1)

        #EdgeConv3
        data3= self.edge_conv3(data2)

        #EdgeConv4
        data4= self.edge_conv4(data3)

        #concatenating data1,2,3,4 together
        data = torch.cat((data1, data2, data3, data4), dim=1)

        ## max pool here
        data = self.conv1(data)
        data5 = F.adaptive_max_pool1d(data,1).view(batch_size, -1)
        data6 = F.adaptive_avg_pool1d(data,1).view(batch_size, -1)
        data= torch.cat((data5,data6),1)

        #Generating the classification score
        data = self.fc1(data)
        data = self.dp1(data)

        data = self.fc2(data)
        data = self.dp2(data)

        data = self.fc3(data)

        return data
