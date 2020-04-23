from __future__ import print_function
import numpy as np
from psnr import psnr
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from model import _netG

from PIL import Image, ImageFilter,ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/val/',
                    help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int,
                    default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='model/netG_streetview.pth',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--outf', default='.',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int, default=4000,
                    help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred', type=int,
                    default=4, help='overlapping edges')
parser.add_argument('--nef', type=int, default=64,
                    help='of encoder filters in first conv layer')
parser.add_argument('--wtl2', type=float, default=0.999,
                    help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)


netG = _netG(opt)
netG.load_state_dict(torch.load(
    opt.netG, map_location=lambda storage, location: storage)['state_dict'])
netG.eval()

transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(
    opt.batchSize, 3, opt.imageSize, opt.imageSize)
real_center = torch.FloatTensor(
    opt.batchSize, 3, int(opt.imageSize/2), int(opt.imageSize/2))

criterionMSE = nn.MSELoss()

if opt.cuda:
    netG.cuda()
    input_real, input_cropped = input_real.cuda(), input_cropped.cuda()
    criterionMSE.cuda()
    real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
real_center = Variable(real_center)

dataiter = iter(dataloader)
real_cpu, _ = dataiter.next()

input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
real_center_cpu = real_cpu[:, :, int(opt.imageSize/4):int(opt.imageSize/4) +
                           int(opt.imageSize/2), int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

#TODO: preprocessing the images
mode = 2
recon_image = None
#DA = DataAugmentation.DataAugmentation()
#input_cropped_edited = None
if (mode ==1): #decolor
    print(input_cropped.shape)
    input_cropped_edited = [transforms.ToPILImage()(x) for x in input_cropped]
    input_cropped_edited = [ImageOps.grayscale(x) for x in input_cropped_edited]
    #input_cropped_edited = [transforms.Grayscale()(x) for x in input_cropped_edited]
    input_cropped_edited = [transforms.ToTensor()(x) for x in input_cropped_edited]
    input_cropped_edited = torch.stack(input_cropped_edited)
    print(input_cropped_edited.shape)
    recon_image = input_cropped_edited.clone()
    for i in range(3):
        input_cropped[:,i,:,:] = input_cropped_edited[:,0,:,:]
    #vutils.save_image(input_cropped,'./data_augmentation/decolor/cut1.png')
elif (mode==2): #flip
    input_cropped= torch.flip(input_cropped,dims=[3])
    print(input_cropped.shape)
    #vutils.save_image(real_cpu[0], 'mode_2_orgi.png', normalize=True)
    #vutils.save_image(input_cropped[0],'mode_2_flipped.png',normalize=True)
elif (mode==3): #Gaussian filter
    input_cropped_edited = [transforms.ToPILImage()(x) for x in input_cropped]
    input_cropped_edited = [x.filter(ImageFilter.GaussianBlur(radius=3)) for x in input_cropped_edited]
    input_cropped_edited = [transforms.ToTensor()(x) for x in input_cropped_edited]
    input_cropped_edited = torch.stack(input_cropped_edited)
    input_cropped = input_cropped_edited
elif (mode==4): #Median filter
    input_cropped_edited = [transforms.ToPILImage()(x) for x in input_cropped]
    input_cropped_edited = [x.filter(ImageFilter.MedianFilter(size=3))for x in input_cropped_edited]
    input_cropped_edited = [transforms.ToTensor()(x) for x in input_cropped_edited]
    input_cropped_edited = torch.stack(input_cropped_edited)
    input_cropped = input_cropped_edited
elif (mode==5): #Egde enhance
    input_cropped_edited = [transforms.ToPILImage()(x) for x in input_cropped]
    input_cropped_edited = [x.filter(ImageFilter.EDGE_ENHANCE)for x in input_cropped_edited]
    input_cropped_edited = [transforms.ToTensor()(x) for x in input_cropped_edited]
    input_cropped_edited = torch.stack(input_cropped_edited)
    input_cropped = input_cropped_edited
elif (mode==6):#rotate
    input_cropped= torch.rot90(input_cropped,dims=[2,3])
    print(input_cropped.shape)

#crop the image
input_cropped.data[:, 0, int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred,
                   int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred] = 2*117.0/255.0 - 1.0
input_cropped.data[:, 1, int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred,
                   int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred] = 2*104.0/255.0 - 1.0
input_cropped.data[:, 2, int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred,
                   int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred] = 2*123.0/255.0 - 1.0


vutils.save_image(real_cpu[0], 'mode_orgi.png', normalize=True)
vutils.save_image(input_cropped[0],'mode_flipped.png',normalize=True)

'''
rgb_weights = torch.FloatTensor([0.2989, 0.5870, 0.1140]).resize_(torch.Size([3,1]))
for i in range(100):
    temp_size = input_cropped.data[i,:,:,:].size()
    temp_size2 = torch.Size([128,128,3])
    m = input_cropped.data[i,:,:,:].resize_(temp_size2)
    print(m.size(),rgb_weights.size())
    m = torch.mm(input_cropped.data[i,:,:,:],rgb_weights)
    input_cropped.data[i,:,:,:].resize_(t)
    
    t = np.copy(input_cropped.data[i,:,:,:])
    t = np.reshape(t,(128,128,3))
    t= np.dot(t,rgb_weights)
    input_cropped.data[i,:,:,:] = np.reshape(t,(3, 128,128)) 
'''
    
#call the model and run
print(input_cropped.shape)
#vutils.save_image(input_cropped.data,
#                  'val_cropped_samples.png', normalize=True)
fake = netG(input_cropped)
errG = criterionMSE(fake, real_center)
if (mode !=1):
    recon_image = input_cropped.clone()
#original
#recon_image.data[:, :, int(opt.imageSize/4):int(opt.imageSize/4) + int(opt.imageSize/2),
#                 int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)] = fake.data
#edited for decolor
#fake.data[:,0,:,:] += fake.data[:,1,:,:]+fake.data[:,2,:,:]

if (mode!=1):
    recon_image.data[:, :, int(opt.imageSize/4):int(opt.imageSize/4) + int(opt.imageSize/2),
                 int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)] = fake.data
file_path = ''
if (mode==1):
    fake_edited = [transforms.ToPILImage()(x) for x in fake.data]
    fake_edited = [ImageOps.grayscale(x) for x in fake_edited]
    #input_cropped_edited = [transforms.Grayscale()(x) for x in input_cropped_edited]
    fake_edited = [transforms.ToTensor()(x) for x in fake_edited]
    fake_edited = torch.stack(fake_edited)
    
    recon_image.data[:, :, int(opt.imageSize/4):int(opt.imageSize/4) + int(opt.imageSize/2),
                 int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)] = fake_edited
    #vutils.save_image(recon_image,'./data_augmentation/decolor/fake.png')
    
    #recon_image=recon_image[:,0,:,:].resize_(torch.Size([100, 1, 128, 128]))
    file_path = './data_augmentation/decolor/'
elif(mode==2):
    file_path = './data_augmentation/flip/'
elif(mode==3):
    file_path = './data_augmentation/guassian/'
elif(mode==4):
    file_path = './data_augmentation/medianFilter/'
elif(mode==5):
    file_path = './data_augmentation/edgeEnhance/'
elif(mode==6):
    file_path = './data_augmentation/rotate/'
vutils.save_image(fake.data[0],'center.png',normalize=True)
vutils.save_image(real_cpu, file_path+'val_real_samples.png', normalize=True)
vutils.save_image(input_cropped.data,
                  file_path+'val_cropped_samples.png', normalize=True)
vutils.save_image(recon_image.data, file_path+'val_recon_samples.png', normalize=True)
p = 0
l1 = 0
l2 = 0
if (mode==6):
    fake = torch.rot90(fake,dims=[2,3])
    fake = torch.rot90(fake,dims=[2,3])
    fake = torch.rot90(fake,dims=[2,3])
if (mode==2):
    fake= torch.flip(fake,dims=[3])
fake = fake.data.numpy()
real_center = real_center.data.numpy()

t = real_center - fake

l2 = np.mean(np.square(t))
l1 = np.mean(np.abs(t))
real_center = (real_center+1)*127.5
fake = (fake+1)*127.5

for i in range(opt.batchSize):
    p = p + psnr(real_center[i].transpose(1, 2, 0), fake[i].transpose(1, 2, 0))

#print results session
if(mode==1):
    print("Decolor the image!")
elif (mode==2):
    print("Flip the image!")
elif (mode==3):
    print('Guassian blur the image!')
elif (mode==4):
    print('Median fliter!')
elif (mode==5):
    print('Edge enhancement!')
elif (mode==6):
    print('Rotate the image!')
print("l2: ",l2)

print("l1: ",l1)

print(p/opt.batchSize)
