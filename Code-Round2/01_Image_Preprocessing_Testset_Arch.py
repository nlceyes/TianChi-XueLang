# -*- coding: utf-8 -*-
# 根据官方提供答案将上一轮的测试集归档为新训练集
import os, shutil
import time

###
category_list = ['正常', '扎洞', '毛斑', '擦洞', '毛洞', '织稀', '吊经', '缺经', '跳花', '油渍', '污渍', '其他']
img_folders = ['..\\data\\xuelang_round1_test_a_20180709', '..\\data\\xuelang_round1_test_b']
xml_folders = ['..\\data\\xuelang_round1_answer_a_20180808', '..\\data\\xuelang_round1_answer_b_20180808']
output_folder = '..\\data\\temp'

###
def img_archive_xml(img_folder, xml_folder, output_folder):
	category_list_xml = os.listdir(xml_folder)
	category_unique = [val for val in category_list_xml if val in category_list]
	category_merge = list(set(category_list_xml).difference(set(category_list))) 
	for category in category_unique:
		file_list = os.listdir(xml_folder+'\\'+category)
		for file in file_list:
			file_name = file[:-4] + '.jpg'
			shutil.move(img_folder+'\\'+file_name, output_folder+'\\'+category)
			shutil.copy(xml_folder+'\\'+category+'\\'+file, output_folder+'\\'+category)
	for category in category_merge:
		if category == '.DS_Store':
			pass
		else:
			file_list = os.listdir(xml_folder+'\\'+category)
			for file in file_list:
				file_name = file[:-4] + '.jpg'
				shutil.move(img_folder+'\\'+file_name, output_folder+'\\'+'其他')
				shutil.copy(xml_folder+'\\'+category+'\\'+file, output_folder+'\\'+'其他')
	for img_file in os.listdir(img_folder):
		shutil.move(img_folder+'\\'+img_file, output_folder+'\\'+'正常')

###
t0 = time.time()
# 初始化temp目录
if not os.path.exists('..\\data\\temp'):
	os.mkdir('..\\data\\temp')
	for category in category_list:
		if not os.path.exists('..\\data\\temp\\'+category):
			os.mkdir('..\\data\\temp\\'+category)
for img_folder, xml_folder in zip(img_folders, xml_folders):
	print('Processing... %s' % img_folder)
	img_archive_xml(img_folder, xml_folder, output_folder)
print('Done in %.3f s.' %(time.time()-t0))