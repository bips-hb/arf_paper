#Standard cDCGAN

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()
        
        # image input , input size : (batch_size, 1, 28, 28)
        self.layer_x = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
                                            kernel_size=4, stride=2, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )
                                    
        # label input, input size : (batch_size, 10, 28, 28)
        self.layer_y = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=32,
                                            kernel_size=4, stride=2, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )
        
        # concat image layer and label layer, input size : (batch_size, 64, 14, 14)
        self.layer_xy = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128,
                                                kernel_size=4, stride=2, padding=1, bias=False),
                                # out size : (batch_size, 128, 7, 7)
                                nn.BatchNorm2d(128),
                                # out size : (batch_size, 128, 7, 7)
                                nn.LeakyReLU(0.2, inplace=True),
                                # out size : (batch_size, 128, 7, 7)
                                nn.Conv2d(in_channels=128, out_channels=256,
                                            kernel_size=3, stride=2, padding=0, bias=False),
                                # out size : (batch_size, 256, 3, 3)
                                nn.BatchNorm2d(256),
                                # out size : (batch_size, 256, 3, 3)
                                nn.LeakyReLU(0.2, inplace=True),
                                # out size : (batch_size, 256, 3, 3)
                                nn.Conv2d(in_channels=256, out_channels=1,
                                            kernel_size=3, stride=1, padding=0, bias=False),
                                # out size : (batch_size, 1, 1, 1)
                                # sigmoid layer to convert in [0,1] range
                                nn.Sigmoid()
                                )
  
    def forward(self, x, y):
        # size of x : (batch_size, 1, 28, 28)
        x = self.layer_x(x)
        # size of x : (batch_size, 32, 14, 14)
        
        # size of y : (batch_size, 10, 28, 28)
        y = self.layer_y(y)
        # size of y : (batch_size, 32, 14, 14)
        
        # concat image layer and label layer output
        xy = torch.cat([x,y], dim=1)
        # size of xy : (batch_size, 64, 14, 14)
        xy = self.layer_xy(xy)
        # size of xy : (batch_size, 1, 1, 1)
        xy = xy.view(xy.shape[0], -1)
        # size of xy : (batch_size, 1)
        return xy

class Generator(nn.Module):

    def __init__(self, input_size=100):

        super(Generator, self).__init__()

        self.input_size = input_size

        # noise z input layer : (batch_size, 100, 1, 1)
        self.layer_x = nn.Sequential(nn.ConvTranspose2d(in_channels=input_size, out_channels=128, kernel_size=3,
                                                    stride=1, padding=0, bias=False),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.BatchNorm2d(128),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.ReLU(),
                                    # out size : (batch_size, 128, 3, 3)
                                    )
        
        # label input layer : (batch_size, 10, 1, 1)
        self.layer_y = nn.Sequential(nn.ConvTranspose2d(in_channels=10, out_channels=128, kernel_size=3,
                                                    stride=1, padding=0, bias=False),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.BatchNorm2d(128),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.ReLU(),
                                    # out size : (batch_size, 128, 3, 3)
                                    )
        
        # noise z and label concat input layer : (batch_size, 256, 3, 3)
        self.layer_xy = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                                                    stride=2, padding=0, bias=False),
                                # out size : (batch_size, 128, 7, 7)
                                nn.BatchNorm2d(128),
                                # out size : (batch_size, 128, 7, 7)
                                nn.ReLU(),
                                # out size : (batch_size, 128, 7, 7)
                                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                                    stride=2, padding=1, bias=False),
                                # out size : (batch_size, 64, 14, 14)
                                nn.BatchNorm2d(64),
                                # out size : (batch_size, 64, 14, 14)
                                nn.ReLU(),
                                # out size : (batch_size, 64, 14, 14)
                                nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,
                                                    stride=2, padding=1, bias=False),
                                # out size : (batch_size, 1, 28, 28)
                                nn.Tanh())
                                # out size : (batch_size, 1, 28, 28)
    
    def forward(self, x, y):
        # x size : (batch_size, 100)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # x size : (batch_size, 100, 1, 1)
        x = self.layer_x(x)
        # x size : (batch_size, 128, 3, 3)
        
        # y size : (batch_size, 10)
        y = y.view(y.shape[0], y.shape[1], 1, 1)
        # y size : (batch_size, 100, 1, 1)
        y = self.layer_y(y)
        # y size : (batch_size, 128, 3, 3)
        
        # concat x and y 
        xy = torch.cat([x,y], dim=1)
        # xy size : (batch_size, 256, 3, 3)
        xy = self.layer_xy(xy)
        # xy size : (batch_size, 1, 28, 28)
        return xy

# custom weights initialization
def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)


class cDCGAN():

    def __init__(self, NUM_EPOCH = 20, BATCH_SIZE=128, size_z=100, Ksteps=1, Adam_lr = 0.0002, Adam_beta1= 0.5, Adam_beta2 = 0.999):

        self.NUM_EPOCH = NUM_EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.size_z = size_z
        self.Ksteps = Ksteps
        self.Adam_lr = Adam_lr
        self.Adam_beta1 = Adam_beta1
        self.Adam_beta2 = Adam_beta2

        self.data = None
        self.conditionalVec = None
        self.netD = None
        self.netG = None
        
    def fit(self, data, conditionalVec, parallelizeGPUs=False):
        
        self.data = data
        self.conditionalVec = conditionalVec
        n = self.data.shape[0]
        p = self.data.shape[1]
        img_size = int(np.sqrt(p))
        
        indices = list(range(n))
        steps = n // self.BATCH_SIZE

        # use cuda if available
        DEVICE = torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")

        train_data = torch.tensor(data.values,dtype=torch.float).to(DEVICE)
        train_labels = torch.tensor(conditionalVec.values,dtype=torch.long).to(DEVICE)

        # labels for training images x for Discriminator training
        labels_real = torch.ones((self.BATCH_SIZE, 1)).to(DEVICE)
        # labels for generated images G(z) for Discriminator training
        labels_fake = torch.zeros((self.BATCH_SIZE, 1)).to(DEVICE)
        # convert labels to onehot encoding
        onehot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)
        # reshape labels to image size, with number of labels as channel
        fill = torch.zeros([10, 10, img_size, img_size])
        #channel corresponding to label will be set one and all other zeros
        for i in range(10):
            fill[i, i, :, :] = 1

        # Create Discriminator and Generator
        if not parallelizeGPUs:
            self.netD = Discriminator().to(DEVICE)
            self.netG = Generator().to(DEVICE)
        else:
            self.netD = Discriminator()
            self.netG = Generator()
            self.netD = nn.DataParallel(self.netD).to(DEVICE)
            self.netG = nn.DataParallel(self.netG).to(DEVICE)

        # randomly initialize all weights to mean=0, stdev=0.2.
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)
        # We calculate Binary cross entropy loss
        criterion = nn.BCELoss()
        # Adam optimizer for generator 
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.Adam_lr, betas=(self.Adam_beta1, self.Adam_beta2))
        # Adam optimizer for discriminator 
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.Adam_lr, betas=(self.Adam_beta1, self.Adam_beta2))

        # number of training steps done on discriminator 
        for epoch in range(self.NUM_EPOCH):
            print('epoch:'+str(epoch))
            shuffledIndices = indices.copy()
            random.shuffle(shuffledIndices)
            shuffledIndices = sorted(shuffledIndices)
            # iterate through data loader generator object
            for step in range(steps):
                range_start = step * self.BATCH_SIZE
                range_end = (step+1) * self.BATCH_SIZE
                
                ############################
                # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                x = train_data[shuffledIndices[range_start:range_end]].to(DEVICE)
                x = x.reshape(self.BATCH_SIZE,1,img_size,img_size)
                # preprocess labels for feeding as y input
                # D_y shape will be (batch_size, 10, 28, 28)
                y_labels = train_labels[shuffledIndices[range_start:range_end]].to(DEVICE)
                D_y = fill[y_labels].to(DEVICE)
                # forward pass D(x)
                x_preds = self.netD(x, D_y)
                # calculate loss log(D(x))
                D_x_loss = criterion(x_preds, labels_real)
                
                # create latent vector z from normal distribution 
                z = torch.randn(self.BATCH_SIZE, self.size_z).to(DEVICE)
                # create random y labels for generator
                y_gen = (torch.rand(self.BATCH_SIZE, 1)*10).type(torch.LongTensor).squeeze()
                # convert genarator labels to onehot
                G_y = onehot[y_gen].to(DEVICE)
                # preprocess labels for feeding as y input in D
                # DG_y shape will be (batch_size, 10, 28, 28)
                DG_y = fill[y_gen].to(DEVICE)
                
                # generate image
                fake_image = self.netG(z, G_y)
                # calculate D(G(z)), fake or not
                z_preds = self.netD(fake_image.detach(), DG_y)
                # loss log(1 - D(G(z)))
                D_z_loss = criterion(z_preds, labels_fake)
                
                # total loss = log(D(x)) + log(1 - D(G(z)))
                D_loss = D_x_loss + D_z_loss
                
                # zero accumalted grads
                self.netD.zero_grad()
                # do backward pass
                D_loss.backward()
                # update discriminator model
                optimizerD.step()
                
                ############################
                # Update G network: maximize log(D(G(z)))
                ###########################
                    
                # if Ksteps of Discriminator training are done, update generator
                if step % self.Ksteps == 0:
                    # As we done one step of discriminator, again calculate D(G(z))
                    z_out = self.netD(fake_image, DG_y)
                    # loss log(D(G(z)))
                    G_loss = criterion(z_out, labels_real)
                    # zero accumalted grads
                    self.netG.zero_grad()
                    # do backward pass
                    G_loss.backward()
                    # update generator model
                    optimizerG.step()
            else:
                # set generator to evaluation mode
                self.netG.eval()
                self.netG.train()
    
    def sample(self,n,conditionalVec=None):
        
        # use cuda if available
        DEVICE = torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")
        
        p = self.data.shape[1]
        img_size = int(np.sqrt(p))
        size_z = self.size_z

        onehot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)

        # create latent vector z from normal distribution 
        z = torch.randn(n, size_z).to(DEVICE)
        
        # create random y labels for generator
        if conditionalVec is None:
            y_gen = (torch.rand(n, 1)*10).type(torch.LongTensor).squeeze()
        else:
            y_gen = torch.tensor(conditionalVec.values)
        # convert genarator labels to onehot
        G_y = onehot[y_gen].to(DEVICE)
        fake_image = self.netG.forward(z,G_y)
        fake_image = pd.DataFrame(fake_image.reshape(n,img_size**2).to('cpu').detach().numpy())
        fake_image.columns = self.data.columns
        fake_image[fake_image<0] = 0
        fake_image[fake_image>0] = 1
        label = pd.DataFrame(G_y.to('cpu').numpy().argmax(axis=1))
        label.columns = ['label']
        return (pd.concat((fake_image,label), axis=1)).reset_index(drop=True)