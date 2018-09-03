# -*- coding: utf-8 -*-
# 准备测试集
import os, shutil
import time
from PIL import Image
from PIL import ImageFile # 解决部分图片无法处理
ImageFile.LOAD_TRUNCATED_IMAGES = True # 解决部分图片无法处理

pixel = 224 # ResNet:224, Inception:299, NasNet:331 
testset_folder = '..\\data\\xuelang_round2_test_a_20180809'

# 初始化test目录
if not os.path.exists('..\\data\\test'):
	os.mkdir('..\\data\\test')
# 调整分辨大小
t0 = time.time()
Image_list = os.listdir(testset_folder)
Image_count = 0
for f in Image_list:
	if f.split('.')[-1] == 'jpg':
		img = Image.open(testset_folder+'\\'+f)
		img_resized = img.resize((pixel, pixel), Image.ANTIALIAS)
		img_resized.save('..\\data\\test\\'+f)
		Image_count += 1
print('Testset of %d pics Prepared in %.3f s.' %(Image_count, time.time()-t0))