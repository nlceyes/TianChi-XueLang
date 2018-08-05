# -*- coding: utf-8 -*-
# Transfer Learning of ResNet 
# Cur Rev.1: ResNet18: batch_size=32, EPOCH=20, lr=0.0001, lr_step = 5, gamma=0.5
import datetime
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
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

# 读取彩色图片并转换为Pytorch格式的图像矩阵
def img2matrix(file_address):
	img = Image.open(file_address)
	img_array_numpy = np.asarray(img, 'uint8') # RGB三通道各像素值0-255(uint8)
	img_array_pytorch = np.transpose(img_array_numpy, (2, 0, 1)) # 转置依据维度的索引！
	return img_array_pytorch

# 构建训练集及其标签
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

# 构建测试集及其标签(假设为1, 因为TensorDataset需要target!)
def testset_and_labels(folder_address):
	pic_list = os.listdir(folder_address + '\\test')
	pic_num = len(pic_list)
	pic_filenames = []
	pic_labels = []
	pic_matrix = np.zeros((pic_num, 3, 224, 224), 'uint8') # 224根据像素手动调整！
	for i in range(pic_num):
		file_name = pic_list[i]
		pic_filenames.append(file_name)
		pic_labels.append('1')
		pic_matrix[i] = img2matrix(folder_address + '\\test\\%s' % file_name)
	return pic_matrix, pic_labels, pic_filenames

# 数据标准化
def custom_normalization(data, std, mean):
	return (data - mean) / std

# 训练模型
def train(batch_size, epoch, train_loader, model, criterion, optimizer):
	model.train()
	print('Current LR:', optimizer.state_dict()['param_groups'][0]['lr'])
	train_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data).cuda(), Variable(target).cuda()
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		train_loss += loss.data[0]
		loss.backward()
		optimizer.step()
	print('Train Epoch: {}   \tLoss: {:.3f}'.format(epoch, train_loss))
	return train_loss

# 测试模型
def test(test_loader, model):
	model.eval()
	test_pred = np.zeros((0,))
	test_proba = np.zeros((0,))
	for data, target in test_loader:
		data, target = Variable(data, volatile = True).cuda(), Variable(target).cuda()
		output = model(data)
		pred = torch.max(output.data, 1)[1].cpu().numpy() # 获得最大概率所对应的标签
		proba = F.softmax(output, 1).data[:, 1].cpu().numpy() # 获得缺陷的Softmax概率
		test_pred = np.hstack((test_pred, pred))
		test_proba = np.hstack((test_proba, proba))
	return test_pred, test_proba

######
t0 = time.time()
#数据集: 训练集和测试集
X, y = dataset_and_labels('..\\data')
Xt, yt, zt = testset_and_labels('..\\data')
y, yt = np.asarray(y).astype(int), np.asarray(yt).astype(int)

# 数据标准化
mean, std = X.mean(), X.std()
print(mean)
print(std)
X = custom_normalization(X, mean, std)
Xt = custom_normalization(Xt, mean, std)

# 将numpy数据转为张量，并构建pytorch数据集
train_x, train_y = torch.from_numpy(X).float(), torch.from_numpy(y)
test_x, test_y = torch.from_numpy(Xt).float(), torch.from_numpy(yt)
train_dataset = TensorDataset(data_tensor = train_x, target_tensor = train_y)
test_dataset = TensorDataset(data_tensor = test_x, target_tensor = test_y)
train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = 64) # batch_size根据情况修改, shuffle=False!

# 建立神经网络并指定优化算法和误差函数
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.cuda()
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # lr=0.0001, betas=(0.9,0.999), eps=1e-08, weight_decay=0
scheduler = lr_scheduler.StepLR(optimizer, lr_step, 0.5) # step_size=5, gamma=0.5

# RUN!
if __name__ == '__main__':
	# torch.set_num_threads(8) # For CPU!
	epoch_list = []
	loss_list = []
	EPOCH = EPOCH
	for epoch in range(1, EPOCH + 1):
		epoch_list.append(epoch)
		scheduler.step()
		loss_data = train(batch_size, epoch, train_loader, model, criterion, optimizer)
		loss_list.append(loss_data)
	torch.save(model, 'model_ResNet18_E20.pth')
	preds, probas = test(test_loader, model)
	df = pd.DataFrame()
	df['filename'] = zt
	# df['prediction'] = preds
	df['probability'] = probas
	date_flag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	df.to_csv('..\\submit\\submit_' + date_flag + '_ResNet18_E20.csv', index=False)
	print('\nAll Done in %.3f mins\n' % ((time.time() - t0) / 60)) # second to minute
plt.plot(epoch_list, loss_list, 'o-', color='r', label='loss')
plt.show()