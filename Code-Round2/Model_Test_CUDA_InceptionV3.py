# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# 读取彩色图片并转换为Pytorch格式的图像矩阵
def img2matrix(file_address):
	img = Image.open(file_address)
	img_array_numpy = np.asarray(img, 'uint8') # RGB三通道各像素值0-255(uint8)
	img_array_pytorch = np.transpose(img_array_numpy, (2, 0, 1)) # 转置依据维度的索引！
	return img_array_pytorch

# 构建测试集及其标签(假设为1, 因为TensorDataset需要target!)
def testset_and_labels(folder_address):
	pic_list = os.listdir(folder_address + '\\test_299')
	pic_num = len(pic_list)
	pic_filenames = []
	pic_labels = []
	pic_matrix = np.zeros((pic_num, 3, 299, 299), 'uint8') # 299根据像素手动调整！
	for i in range(pic_num):
		file_name = pic_list[i]
		pic_filenames.append(file_name)
		pic_labels.append('1')
		pic_matrix[i] = img2matrix(folder_address + '\\test_299\\%s' % file_name)
	return pic_matrix, pic_labels, pic_filenames

# 数据标准化
def custom_normalization(data, std, mean):
	return (data - mean) / std

# 测试模型
def test(test_loader, model):
	model.eval()
	test_pred = np.zeros((0,))
	test_proba = np.zeros((0, 11))
	for data, target in test_loader:
		data, target = Variable(data, volatile = True).cuda(), Variable(target).cuda()
		output = model(data)
		pred = torch.max(output.data, 1)[1].cpu().numpy() # 获得最大概率所对应的标签
		proba = F.softmax(output, 1).data.cpu().numpy() # 获得Softmax概率
		test_pred = np.hstack((test_pred, pred))
		test_proba = np.vstack((test_proba, proba))
	return test_pred, test_proba

######
t0 = time.time()
#数据集: 测试集
Xt, yt, zt = testset_and_labels('..\\data')
yt = np.asarray(yt).astype(int)
# 数据标准化
# 均值和标准差根据训练集数据修改！！
# mean, std = 134.1789, 21.8115 # trainset of 3331 pics (224x224)
mean, std = 134.18, 22.08 # trainset of 3331 pics (299x299)
print(mean)
print(std)
Xt = custom_normalization(Xt, mean, std)
# 将numpy数据转为张量，并构建pytorch数据集
test_x, test_y = torch.from_numpy(Xt).float(), torch.from_numpy(yt)
test_dataset = TensorDataset(data_tensor = test_x, target_tensor = test_y)
test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = 8) # batch_size根据情况修改, shuffle=False!
# 载入训练网络并预测
model = torch.load('..\\models\\Round2\\model_InceptionV3_B16_E06_3.pth') # 根据模型修改！
model.cuda()
# print(model)
preds, probas = test(test_loader, model)
# 写入结果文件
df = pd.DataFrame()
df['filename'] = zt
df['pred'] = preds
for i in range(11):
	df['proba_'+str(i)] = probas[:, i]
df.to_csv('InceptionV3_B16_Preds_E06_3_Ori.csv', index=False) # 根据模型修改！！
spliter = '|'
file_list = df['filename']
filename_defect_list = np.zeros((0,))
probability_list = np.zeros((0,))
for i in range(11):
	if i == 0:
		prefix = 'norm'
	else:
		prefix = 'defect_' + str(i)
	filename_defect = file_list + spliter + prefix
	filename_defect_list = np.hstack((filename_defect_list, filename_defect))
	probability = df['proba_'+str(i)]
	probability_list = np.hstack((probability_list, probability))
df_rev = pd.DataFrame()
df_rev['filename|defect'] = filename_defect_list
df_rev['probability'] = probability_list
df_rev.to_csv('InceptionV3_B16_Preds_E06_3_Rev.csv', index=False, float_format = '%.5f')
print('\nAll Done in %.3f mins\n' % ((time.time() - t0) / 60)) # second to minute