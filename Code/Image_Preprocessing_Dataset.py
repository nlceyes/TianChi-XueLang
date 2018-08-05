# -*- coding: utf-8 -*-
# 图片预处理：分辨率大小(224)和类型(无瑕疵:0, 有瑕疵:1)
import os
from PIL import Image
from PIL import ImageFile # 解决部分图片无法处理
ImageFile.LOAD_TRUNCATED_IMAGES = True # 解决部分图片无法处理

pixel = input('Please input your desired pixel:\n>>>')
pic_type = input('Please input the pic type:\n>>>')

Image_list = os.listdir('..\\data')
if not os.path.exists('..\\data\\train'):
	os.mkdir('..\\data\\train')
Image_count = 0

#仅对图片进行预处理，跳过文件夹件和Python脚本
for f in Image_list:
	if len(f.split('.')) == 1:
		pass
	if f.split('.')[-1] == 'py':
		pass
	else:
		img = Image.open('..\\data\\' + f)
		img_resized = img.resize((int(pixel), int(pixel)), Image.ANTIALIAS)
		img_resized.save('..\\data\\train\\' + pic_type + '_' + str(Image_count) + '.jpg')
		Image_count += 1
		os.remove('..\\data\\' + f)

print('Totally %d pictures have been pre-processed!' % Image_count)