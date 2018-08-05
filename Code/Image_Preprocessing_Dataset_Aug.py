# -*- coding: utf-8 -*-
#图片预处理：对原始图片进行水平和垂直翻转，增强训练集
import os, shutil
from PIL import Image
from PIL import ImageFile # 解决部分图片无法处理
ImageFile.LOAD_TRUNCATED_IMAGES = True # 解决部分图片无法处理

shutil.copytree('..\\data\\train', '..\\data\\train_vt')

Image_list = os.listdir('..\\data\\train_vt')
Image_count = 0

#仅对图片进行预处理，跳过文件夹件和Python脚本
for f in Image_list:
	if len(f.split('.')) == 1:
		pass
	if f.split('.')[-1] == 'py':
		pass
	else:
		img = Image.open('..\\data\\train_vt\\' + f)
		img_t = img.transpose(Image.FLIP_LEFT_RIGHT)
		img_t.save('..\\data\\train_vt\\' + f.split('.')[0] + '_t' + '.jpg')
		img_v = img.transpose(Image.FLIP_TOP_BOTTOM)
		img_v.save('..\\data\\train_vt\\' + f.split('.')[0] + '_v' + '.jpg')
		img_t_v = img_t.transpose(Image.FLIP_TOP_BOTTOM)
		img_t_v.save('..\\data\\train_vt\\' + f.split('.')[0] + '_t_v' + '.jpg')
		Image_count += 1

print('Totally %d pictures have been pre-processed!' % Image_count)