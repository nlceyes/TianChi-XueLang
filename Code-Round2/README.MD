# 复赛

0.预处理代码：

	0.1.Image_Preprocessing_Testset_Arch.py

		将初赛测试集AB及其答案解压后拷贝至data目录，运行后生成temp子目录，存放按分类要求后的图片
	
	0.2.Image_Preprocessing_Dataset.py

		运行后生成train子目录，存放训练集(调整分辨率并增加分类标签)
	
	0.3.Image_Preprocessing_Testset.py

		将测试集图片解压拷贝至data目录，运行后生成test子目录，存放调整分辨率后的测试集

1.模型代码：

	1.1.InceptionV3_Run_CUDA.py

		模型1，基于data\train训练集和pytorch自带预训练InceptionV3网络，运行后生成模型及预测结果文件
	
	1.2.NasNet-A-Large_Run_CUDA.py

		模型2，基于data\train训练集和pytorch第三方预训练NasNet网络，运行后生成模型及预测结果文件
	
2.结果后处理：
	
	Model_Test_CUDA_InceptionV3.py and Model_Test_CUDA_NasNet.py
  
		载入1.中模型并生成标准格式的预测结果文件，然后手动取均值最作为最终问结果。

3.ResNet_Test_FL_CUDA.py:
	
	测试Focal Loss

3.心得：

  * 机器的算力非常重要非常重要非常重要！
  
  * 使用ResNet测试了Focal Loss，效果无提升，均采用论文中默认参数，可能由于Alpha, Gamma, LR未调整至最优值
  
  * 对训练集和测试集均进行了增强，效果无提升
  
  * 若进一步增大图片分辨率，或者将最后的平均池化改为最大池化(检测缺陷！)，可能会有所提高
  
  * 前期调整的Pretrained模型和其得分如下：
  
  		No.		Model											AUC and MAP
		1		ResNet18(Pretrained, Tuned LR)					0.6109
		2		ResNet50(Pretrained, Tuned LR)					0.6148
		3		DenseNet201(Pretrained, Tuned LR)				0.6314
		4		DenseNet201(Pretrained, Tuned LR, Aug Testset)	0.6278
		5		InceptionV3(Pretrained, Tuned LR)				0.6448
		6		InceptionV3(Pretrained, Tuned LR, Aug Trainset)	0.6437
		7		InceptionV4(Pretrained, Tuned LR)				0.6403
		8		NasNet-A-Large(Pretrained, Tuned LR)			0.6639
		9		Avg of 8, 5 (Final)								0.6881
