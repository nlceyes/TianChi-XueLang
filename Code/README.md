# 初赛

0.预处理代码：

	0.1.Image_Preprocessing_Dataset.py

		将有瑕疵和无瑕疵两类图片依次拷贝至data目录，运行后生成train子目录，存放调整分辨率并增加类型标签的训练集
	
	0.2.Image_Preprocessing_Dataset_Aug.py

		运行后生成train_vt子目录，存放增强训练集(增加水平和垂直翻转)
	
	0.3.Image_Preprocessing_Testset.py

		将测试集图片拷贝至data目录，运行后生成test子目录，存放调整分辨率后的测试集

1.模型代码：

	1.1.Model_ResNet18_Run_CUDA.py

		模型1，基于data\train训练集和pytorch自带预训练ResNet18网络，运行后生成ResNet18_E20后缀结果文件
	
	1.2.Model_ResNet18_vt_Run_CUDA.py

		模型2，基于data\train_vt增强训练集和pytorch自带预训练ResNet18网络，运行后生成ResNet18_vt_E20后缀结果文件
	
	1.3.Model_ResNet34_Run_CUDA.py

		模型3，基于data\train训练集和pytorch自带预训练ResNet34网络，运行后生成ResNet34_E20后缀结果文件

2.结果后处理：
	
	将1.1-1.3得到的结果文件取平均，得到最终预测结果。

3.心得：

  * CNN中的权重初始化非常重要非常重要非常重要！
  
  * 选择合适的学习率也非常重要！
  
  * 基于Pretrained模型进行调整比自己手写模型再从头训练更高效！ 
  
  * 前期调整的Pretrained模型和其得分如下：
  
  		No.	Model						AUC
		1	AlexNet(KaiMing)				0.7606
		2	ResNet18(Pretrained)				0.8562
		3	ResNet18(Pretrained, Tuned LR)			0.8809
		4	ResNet18(Pretrained, Tuned LR, Aug Trainset)	0.8812
		5	ResNet34(Pretrained, Tuned LR)			0.8842
		6	Avg of 3, 4, 5 (Final)				0.9118
