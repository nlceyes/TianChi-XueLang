# -*- coding: utf-8 -*-
# 准备训练集
import os, shutil
import time
from PIL import Image
from PIL import ImageFile # 解决部分图片无法处理
ImageFile.LOAD_TRUNCATED_IMAGES = True # 解决部分图片无法处理

pixel = 224 # ResNet:224, Inception:299, NasNet:331 
category_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '9', '10']
category_list = ['正常', '扎洞', '毛斑', '擦洞', '毛洞', '织稀', '吊经', '缺经', '跳花', '油渍', '污渍', '其他']
origin_dataset_folder = '..\\data\\origin_dataset'

# 初始化train目录
if not os.path.exists('..\\data\\train'):
	os.mkdir('..\\data\\train')
# 归档原始训练集
t0 = time.time()
category_list_dataset = os.listdir(origin_dataset_folder)
category_unique = [val for val in category_list_dataset if val in category_list]
category_merge = list(set(category_list_dataset).difference(set(category_list)))
for category in category_unique:
	file_list = os.listdir(origin_dataset_folder+'\\'+category)
	for file in file_list:
		shutil.copy(origin_dataset_folder+'\\'+category+'\\'+file, '..\\data\\temp\\'+category)
for category in category_merge:
	file_list = os.listdir(origin_dataset_folder+'\\'+category)
	for file in file_list:
		shutil.copy(origin_dataset_folder+'\\'+category+'\\'+file, '..\\data\\temp\\其他')
print('Original Dataset Archived in %.3f s.' %(time.time()-t0))
# 调整分辨大小并修改文件名称增加分类标签
t0 = time.time()
Image_count_total = 0
for label, category in zip(category_number, category_list):
	Image_count = 0
	Image_list = os.listdir('..\\data\\temp\\'+category)
	for f in Image_list:
		if f.split('.')[-1] == 'jpg':
			img = Image.open('..\\data\\temp\\'+category+'\\'+f)
			img_resized = img.resize((pixel, pixel), Image.ANTIALIAS)
			img_resized.save('..\\data\\train\\'+label+'_'+str(Image_count_total)+'.jpg')
			Image_count += 1
			Image_count_total += 1
	print('Category of %s labeled with %s in %d pics Done.' %(category, label, Image_count))
print('Dataset of %d pics Prepared in %.3f s.' %(Image_count_total, time.time()-t0))