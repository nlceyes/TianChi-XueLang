# -*- coding: utf-8 -*-
# Transfer Learning of ResNet with Focal Loss
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
import torchvision.models as models

# 超参数
batch_size = 32
EPOCH = 20
lr_step = 5
alpha_norm = [0.036, 1.441, 1.853, 0.365, 0.985, 0.819, 0.326, 1.235, 0.707, 1.898, 0.335]

# Focal Loss
class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average
	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)	# N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)
		logpt = F.log_softmax(input)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())
		if self.alpha is not None:
			if self.alpha.type()!=input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)
		loss = -1 * (1-pt)**self.gamma * logpt
		if self.size_average: return loss.mean()
		else: return loss.sum()

# 读取彩色图片并转换为Pytorch格式的图像矩阵
def img2matrix(file_address):
	img = Image.open(file_address)
	img_array_numpy = np.asarray(img, 'uint8') # RGB三通道各像素值0-255(uint8)
	img_array_pytorch = np.transpose(img_array_numpy, (2, 0, 1)) # 转置依据维度的索引！
	return img_array_pytorch

# 构建数据集及其标签
def dataset_and_labels(folder_address):
	pic_list = os.listdir(folder_address + '\\train')
	pic_num = len(pic_list)
	pic_labels = []
	pic_matrix = np.zeros((pic_num, 3, 224, 224), 'uint8') # 224根据像素手动调整！
	for i in range(pic_num):
		file_name = pic_list[i]
		pic_name = file_name.split('.')[0]
		pic_label = pic_name.split('_')[0]
		pic_labels.append(pic_label)
		pic_matrix[i] = img2matrix(folder_address + '\\train\\%s' % file_name)
	return pic_matrix, pic_labels

# 数据标准化
def custom_normalization(data, std, mean):
	return (data - mean) / std

# 训练模型
def train(epoch, train_loader, model, criterion, optimizer):
	model.train()
	print('Current LR:', optimizer.state_dict()['param_groups'][0]['lr'])
	train_loss = 0
	for data, target in train_loader:
		data, target = Variable(data).cuda(), Variable(target).cuda()
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		train_loss += loss.data[0]
		loss.backward()
		optimizer.step()
	print('Train Epoch: {}   \t\tLoss: {:.3f}'.format(epoch, train_loss/48))
	return train_loss/48

# 测试模型
def test(test_loader, model, criterion):
	model.eval()
	test_loss = 0
	test_correct = 0
	for data, target in test_loader:
		data, target = Variable(data, volatile = True).cuda(), Variable(target).cuda()
		output = model(data)
		loss = criterion(output, target).data[0]
		test_loss += loss
		pred = torch.max(output.data, 1)[1] # 获得最大概率所对应的标签
		correct = (pred.cpu().numpy() == target.cpu().data.numpy()).sum()
		test_correct += correct
	total_size = len(test_loader.dataset)
	print('Accuracy: {}/{}({:.2f}%) \tTest loss: {:.3f}\n'.format(
	test_correct, total_size, 100*test_correct/total_size, test_loss/20))
	return test_loss/20, test_correct / total_size

######
t0 = time.time()
#数据集分为训练集和测试集
dataset, labels = dataset_and_labels('..\\data')
X, Xt, y, yt = train_test_split(dataset, labels, test_size = 0.3, random_state=0) # random_state!
y, yt = np.asarray(y).astype(int), np.asarray(yt).astype(int)
#print(X.dtype, Xt.dtype, y.dtype, yt.dtype)

# 数据标准化
mean, std = X.mean().astype(np.float32), X.std().astype(np.float32)
print('Mean Value:', mean, mean.dtype)
print('Standard Deviation:', std, std.dtype)
X = custom_normalization(X, mean, std)
Xt = custom_normalization(Xt, mean, std)

# 将numpy数据转为张量，并构建pytorch数据集
train_x, train_y = torch.from_numpy(X).float(), torch.from_numpy(y)
test_x, test_y = torch.from_numpy(Xt).float(), torch.from_numpy(yt)
train_dataset = TensorDataset(data_tensor = train_x, target_tensor = train_y)
test_dataset = TensorDataset(data_tensor = test_x, target_tensor = test_y)
train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(dataset = test_dataset, shuffle = True, batch_size = 64)

# 建立神经网络并指定优化算法和误差函数
lr_list = [0.000001]
df = pd.DataFrame()
df['epoch'] = list(range(1, EPOCH + 1))
for i in range(len(lr_list)):
	model = models.resnet18(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 11)
	model.cuda()
	# print(model)
	criterion = FocalLoss()
	print('Current LR:', lr_list[i])
	optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[i]) # betas=(0.9,0.999), eps=1e-08, weight_decay=0
	scheduler = lr_scheduler.StepLR(optimizer, lr_step, 0.5)
	# RUN!
	train_loss_list = []
	test_loss_list = []
	accuracy_list = []
	for epoch in range(1, EPOCH + 1):
		scheduler.step()
		train_loss_data = train(epoch, train_loader, model, criterion, optimizer)
		train_loss_list.append(train_loss_data)
		test_loss_data, acc_data = test(test_loader, model, criterion)
		test_loss_list.append(test_loss_data)
		accuracy_list.append(acc_data)
	df['trn_los_' + str(i+1)] = train_loss_list
	df['tst_los_' + str(i+1)] = test_loss_list
	df['acc_' + str(i+1)] = accuracy_list
	print('Progress: %d/%d\n' % (i+1, len(lr_list)))
df.to_csv('ResNet_FL_LC.csv', index=False)
print('\nAll Done in %.3f mins\n' % ((time.time() - t0) / 60)) # second to minute