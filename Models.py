
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x



class U_Net(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=10, s=2):
        super(U_Net, self).__init__()
        self.scale=s
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=1,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(1 * 28 * 28, 2)
        self.dce=dce_loss(num_classes,num_hidden_units)

    def forward(self,x):
        # print("enter U_Net")
        # print("x: ", x.shape)
        # encoding path
        x1 = self.Conv1(x)
        # print("x1: ", x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print("x2: ", x2.shape)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # print("x3: ", x3.shape)
        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)
        # print("x4: ", x4.shape)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)
        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)
        # d4 = self.Up4(x4)
        # print("d4: ", d4.shape)
        # d4 = torch.cat((x3,d4),dim=1)
        # print("d4: ", d4.shape)
        # d4 = self.Up_conv4(d4)
        # print("d4: ", d4.shape)
        d3 = self.Up3(x3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        # print("d3: ", d3.shape)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        # print("d2: ", d2.shape)
        x = self.Conv_1x1(d2)
        # print("d1: ", x.shape)                            #[50, 1, 28, 28]

        x= x.view(-1, 1 * 28 * 28)
        # print("x: ", x.shape)                             #[50, 1152]
        x1 = self.preluip1(self.ip1(x))
        # print("x: ", x1.shape)                            #[50, 2]
        centers,x=self.dce(x1)
        # print("centers: ", centers.shape)                 #[2, 10]
        # print("x: ", x.shape)                             #[50, 10]
        output = F.log_softmax(self.scale*x, dim=1)
        # print("output: ", output.shape)                   #[50, 10]
    
        return x1,centers,x,output

class Net(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=10, s=2):
        super(Net, self).__init__()
        self.scale=s
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, 2)
        self.dce=dce_loss(num_classes,num_hidden_units)
   
    def forward(self, x):
        print("enter net")
        # print("x: ")                                #[50, 1, 28, 28]
        # print(x.shape)
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        # print("x: ", x.shape)                             #[50, 128, 3, 3]
        x= x.view(-1, 128 * 3 * 3)
        # print("x: ", x.shape)                             #[50, 1152]
        x1 = self.preluip1(self.ip1(x))
        # print("x: ", x1.shape)                            #[50, 2]
        centers,x=self.dce(x1)
        # print("centers: ", centers.shape)                 #[2, 10]
        # print("x: ", x.shape)                             #[50, 10]
        output = F.log_softmax(self.scale*x, dim=1)
        # print("output: ", output.shape)                   #[50, 10]
        # print("x1,centers,x,output: ", x1,centers,x,output)
        return x1,centers,x,output


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes,feat_dim,init_weight=True):
   
        super(dce_loss, self).__init__()
        self.n_classes=n_classes
        self.feat_dim=feat_dim
        self.centers=nn.Parameter(torch.randn(self.feat_dim,self.n_classes).cuda(),requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

     

    def forward(self, x):
   
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers

        return self.centers, -dist

def regularization(features, centers, labels):
        distance=(features-torch.t(centers)[labels])

        distance=torch.sum(torch.pow(distance,2),1, keepdim=True)

        distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return distance

