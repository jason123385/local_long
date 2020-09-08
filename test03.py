import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import os
from torchvision import transforms
import torch.nn.functional as F
import csv
from torchvision.models import resnet18
from utils import Flatten

################################################################################
# class Blk(nn.Module):
#     def __init__(self,ch_in,ch_out,stride):
#         super(Blk,self).__init__()
#         self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
#         self.btn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
#         self.btn2 = nn.BatchNorm2d(ch_out)
#
#         if ch_in == ch_out:
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_out,ch_out,kernel_size=1,stride=stride)
#             )
#         elif ch_in != ch_out:
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride)
#             )
#
#     def forward(self,x):
#         out = F.relu(self.btn1(self.conv1(x)))
#         out = self.btn2(self.conv2(out))
#         output = out + self.extra(x)
#         output = F.relu(output)
#         return output
#
# class Resnet18(nn.Module):
#     def __init__(self):
#         super(Resnet18,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#
#         self.blk1 = Blk(16,32,2)
#         self.blk2 = Blk(32, 64, 2)
#         self.blk3 = Blk(64, 128, 2)
#         self.blk4 = Blk(128, 128, 2)
#
#         self.out = nn.Linear(128*1*1,5)
#
#     def forward(self,x):
#         out = self.conv(x)
#         out = self.blk1(out)
#         out = self.blk2(out)
#         out = self.blk3(out)
#         out = self.blk4(out)
#         out = F.adaptive_avg_pool2d(out, [1, 1])
#
#         out = out.view(out.size(0),-1)
#         output = self.out(out)
#         return output
#
# net = Resnet18()
# net.load_state_dict(torch.load('net1_params.pkl'))
######################################################################



#######################################################################
# class BlkZr(nn.Module):
#     def __init__(self,ch_in,ch_out,stride):
#         super(BlkZr,self).__init__()
#         self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
#         self.btn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
#         self.btn2 = nn.BatchNorm2d(ch_out)
#
#         self.extra = nn.Sequential()  # 先建一个空的extra
#         if ch_out != ch_in:
#             # [b, ch_in, h, w] => [b, ch_out, h, w]
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(ch_out)
#             )
#
#     def forward(self,x):
#         out = F.relu(self.btn1(self.conv1(x)))
#         out = self.btn2(self.conv2(out))
#         output = out + self.extra(x)
#         output = F.relu(output)
#         return output
#
#
# class Resnet17(nn.Module):
#     def __init__(self):
#         super(Resnet17,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3,8,kernel_size=3,stride=3,padding=0),
#             nn.BatchNorm2d(8),
#         )
#
#         self.blk1 = BlkZr(8,16,3)
#         self.blk2 = BlkZr(16, 32, 3)
#         self.blk3 = BlkZr(32, 64, 2)
#         self.blk4 = BlkZr(64, 128, 2)
#
#         self.out = nn.Linear(128*3*3,5)
#
#     def forward(self,x):
#         out = self.conv(x)
#         out = self.blk1(out)
#         out = self.blk2(out)
#         out = self.blk3(out)
#         out = self.blk4(out)
#
#         out = out.view(out.size(0),-1)
#         output = self.out(out)
#         return output
#
# net = Resnet17()
# net.load_state_dict(torch.load('Resnet17_5.pkl'))

#############################################################################


#################################################################################
# class BlkZr(nn.Module):
#     def __init__(self,ch_in,ch_out,stride):
#         super(BlkZr,self).__init__()
#         self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
#         self.btn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
#         self.btn2 = nn.BatchNorm2d(ch_out)
#
#         self.extra = nn.Sequential()  # 先建一个空的extra
#         if ch_out != ch_in:
#             # [b, ch_in, h, w] => [b, ch_out, h, w]
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(ch_out)
#             )
#
#     def forward(self,x):
#         out = F.relu(self.btn1(self.conv1(x)))
#         out = self.btn2(self.conv2(out))
#         output = out + self.extra(x)
#         output = F.relu(output)
#         return output
#
#
# class Resnet16(nn.Module):
#     def __init__(self):
#         super(Resnet16,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3,16,kernel_size=3,stride=3,padding=0),
#             nn.BatchNorm2d(16),
#         )
#
#         self.blk1 = BlkZr(16,32,3)
#         self.blk2 = BlkZr(32, 64, 3)
#         self.blk3 = BlkZr(64, 128, 2)
#         self.blk4 = BlkZr(128, 256, 2)
#
#         self.out = nn.Linear(256*3*3,5)
#
#     def forward(self,x):
#         out = self.conv(x)
#         out = self.blk1(out)
#         out = self.blk2(out)
#         out = self.blk3(out)
#         out = self.blk4(out)
#
#         out = out.view(out.size(0),-1)
#         output = self.out(out)
#         return output
#
# net = Resnet16()
# net.load_state_dict(torch.load('net3_params.pkl'))
##########################################################################################

trained_net = resnet18(pretrained=True)
net = nn.Sequential(*list(trained_net.children())[:-1],  # [b, 512, 1, 1]
                      Flatten(),  # [b, 512, 1, 1] => [b, 512]
                      nn.Linear(512, 5)
                      )


net.load_state_dict(torch.load('Resnet17_6.pkl'))


pics = []
with open('./pokeman/image.csv','r') as f:
    reader = csv.reader(f)
    for line in reader:
        img,label = line
        pics.append(img)


pics = pics[975:1000]
print(pics)
# pics = os.listdir(PATH)
num = 1

for pic in pics:
    if  pic.endswith('.png') or pic.endswith('.jpg'):
        # pic_path = os.path.join(PATH,pic)
        # # pic_path = pic_path.replace('\\','/')
        # print(pic_path)
        im = plt.imread(pic,0)
        images = Image.open(pic)
        images = images.resize((224,224))
        images = images.convert('RGB')
        transform1 = transforms.ToTensor()
        images = transform1(images)
        transform2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        images = transform2(images)
        images = images.reshape(1, 3, 224, 224)
        outputs = net(images)
        values, indices = outputs.data.max(1)
        plt.subplot(6,5,num)
        plt.imshow(im)
        plt.title('{}'.format(int(indices[0])))
        plt.axis('off')
        num = num + 1
plt.show()