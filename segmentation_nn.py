"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        # self.save_hyperparameters(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.hparams = hparams
        self.size =self.hparams['size']
        self.states = []
        self.down_pooling1 = nn.MaxPool2d(2)
        self.down_pooling2 = nn.MaxPool2d(2)
        self.down_pooling3 = nn.MaxPool2d(2)

        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.size, 3, 1, padding = 1),
            nn.LeakyReLU()
        )       
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.size, 2*self.size, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*self.size, 4*self.size, 3,1,padding=1),
            nn.LeakyReLU()
        )
        self.conv4 =nn.Sequential(
            nn.Conv2d(4*self.size, 5*self.size, 3,1,padding=1),
            nn.LeakyReLU()  
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(9*self.size, 5*self.size, 3,1,padding=1),
            nn.LeakyReLU()           
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(7*self.size, 3*self.size, 3,1,padding=1),
            nn.LeakyReLU()           
        )
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(5*self.size, 5*self.size, 3,1,1),
            nn.LeakyReLU()
        )
        
        self.upconv2 =nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(5*self.size, 5*self.size, 3,1,1),
            nn.LeakyReLU()
        )
        self.upconv3 =nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(3*self.size, 2*self.size, 3,1,1),
            nn.LeakyReLU() 
        )
        self.finalconv = nn.Sequential(
            nn.Conv2d(3*self.size, num_classes, 3,1,1)
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.
        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # x = torch.reshape(x, (3,240,240))
        x = self.conv1(x) #S*240*240
        # print(x.shape)
        self.states.append(x)
        # print(x.shape)
        x = self.down_pooling1(x) #S*120*120
        # print(x.shape)
        x = self.conv2(x) #2S*120*120
        # print("4", x.shape)
        self.states.append(x)
        # print("5", x.shape)
        x = self.down_pooling2(x)#2S*60*60
        # print("6", x.shape)
        x = self.conv3(x) #4S*60*60
        # print("7", x.shape)
        self.states.append(x)
        # print("8", x.shape)
        x = self.down_pooling3(x) #4S*30*30
        # print("8", x.shape)
        x= self.conv4(x) #4S*30*30
        # print("8", x.shape)
        x = self.upconv1(x) #4S*60*60
        # print("8", x.shape)
        x= torch.cat((x, self.states[2]), dim =1)#8S*60*60
        # print("9", x.shape)
        x = self.conv5(x) #4S*60*60
        # print("12", x.shape)
        x = self.upconv2(x) #4S*120*120
        # print("x shape:", x.shape, "self.states shape", self.states[1].shape)
        x= torch.cat((x, self.states[1]), dim =1)#6S*120*120
        x = self.conv6(x) #3S*120*120
        x = self.upconv3(x) #2S*240*240
        x= torch.cat((x, self.states[0]), dim =1)#3S*240*240
        x = self.finalconv(x)
        self.states =[]
        # x = torch.reshape(x, (23, 240,240))
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
