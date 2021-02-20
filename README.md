# 100-Days-Of-AI-Code
100 Days of Artificial Intelligence Coding。人工智能100天

之前推出了一个资源合集 AlphaTree计划，https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/
但是不少小伙伴反馈，很多时候资源太多，学起来无从下手 （无奈脸），往往都在收藏夹里吃灰。 于是打算推出一个 人工智能100天计划，带大家一起学习github上的项目和AI的基础知识。而且针对有的小伙伴英文不好的情况，贴心的我还提供了一些翻译链接，如果没有人翻译过，就提供论文机翻给到大家。

如果喜欢，就star一个吧。

## 为什么要做这样的一个AI100天计划

网上有不少机器学习的计划，也有一些roadmap，但是深度学习的不多，而且大多就是论文阅读列表。对于我们很多人来说，很难坚持下去。 
所以我从图像入手，将一些重要的论文 ，论文翻译，源代码，面试题 按发展时间，领域整合在一起。作为深度学习这部分学习的基础。


除了前两周最基础的网络结构外，根据不同的方向可以衍生出多种打卡模式。
每个同学也可以选择自己喜欢方式来进行。
可以只看知识卡片 刷题
也可以结合论文原文 和翻译 深入理解。
也可以一起coding。由于此计划为基础教程，主要用python 和pytorch 来实现。

CC-BY-NC-SA 知识共享-署名-非商业性-相同方式共享


## 按照什么顺序来刷
本计划持续时间约4个月。

![basic](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/ObjectClassification.png)

### 第一个月 深度学习基础 
- [Object Classification 物体分类] 
- A 1-2周 深度学习基本模型
  - [LeNet] 
  - [AlexNet]
  - [NIN]
  - [VGG]
  - [GoogLeNet]
  

  - [ResNet] 
  - [Inception V3]
  - [Inception V4]
  - [Inception-Resnet-V2]
  - [DenseNet]

- B 3周 轻量级模型 & 剪枝
  - [SqueezeNet]
  - [MobileNet]
  - [MobileNetV2]
  - [Xception]
  - [SuffleNet]

- C 4周 Resnet模型变种的发展
  - [ResNeXt]
  - [DPN]
  - [WRN]
  - [PolyNet]
  - [NasNet]
  
![basic1](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/ObjectDetection%26Seg.png)

### 第二个月 物体检测和分割

- [Object Detection 物体检测](#object-detection-物体检测)
- D 1-2周 物体检测
  - [RCNN]
  - [SppNet]
  - [FastRCNN]
  - [FasterRCNN]
  - [Yolo]
  

  - [SSD]
  - [FPN]
  - [RFCN]
  - [RetinaNet]
  - [MaskRCNN]

- [Object Segmentation 物体分割](#object-segmentation-物体分割)
- E 3-4周 物体分割
  - [FCN]
  - [SegNet]
  - [UNet]
  - [DecovNet]
  - [LinkNet]
  - [RefineNet]
  - [PSPNet]
  - [DeepLab]
  - [DenseASPP]
  - [FastFCN]



### 第三个月 与人相关的部分
![face](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/Face.png)
- [Face Detection & Recognition 人脸检测与识别]
- F 1周 人脸与特征点检测
  - [MTCNN](#mtcnn-详解-detail-zhang-kaipeng-乔宇-qiao-yu--cuhk-mmlab--siat)
  - [Deep Face](#deep-face)
  - [FaceNet](#facenet-详解-detail)

[!Pose](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/Pose.png)
- G 2-3周 肢体检测

[!OCR](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/OCR.png)

- [OCR](ocroptical-character-recognition-字符识别--str-scene-text-recognition-场景文字识别)
- H 4周 OCR相关
  - [CTPN](#ctpn-connectionist-text-proposal-network--详解-detail--zhi-tian--乔宇-qiao-yu--cuhk-mmlab--siat)
  - [TextBoxes](#textboxes--详解-detail-白翔-xiang-baimedia-and-communication-lab-hust)
  - [CRNN](#crnn-详解-detail-白翔-xiang-baimedia-and-communication-lab-hust)


![GAN](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/Art%26Ganpic.png)

### 第四个月 GAN 生成对抗网络
- [GAN 生成式对抗网络](#gan-生成式对抗网络)
- I 1-2周 GAN基础
  - [GAN]
  - [CGAN]
  - [InfoGAN]
  - [SemiGAN]
  - [ACGAN]

  - [DCGAN]
  - [ProGAN]
  - [SAGAN]
  - [BigGAN]
  - [StyleGAN]

J 3-4周 GAN应用

![basic2](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/mapclean_1.5.png)




## Day1-15 [Object Classification 物体分类] 深度学习基本模型

### Day1 LeNet

- 知识卡片  / 结构大图  / 论文网络结构图
 
- 简介:
LeNet-5是深度学习最简单的体系结构之一。大量深度学习网络结构的起点。 
它具有2个卷积层和3个完全连接层。

- 特点：
这种架构已成为标准的“模板”：堆叠具有激活功能的卷积层，pooling层，最后加上一个或多个完全连接的FC层。

- 基本信息
论文：基于梯度的学习在文档识别中的应用
paper ：LeCun, Yann; Léon Bottou; Yoshua Bengio; Patrick Haffner (1998). "Gradient-based learning applied to document recognition"
原文：http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
作者：Yann LeCun，LéonBottou，Yoshua Bengio和Patrick Haffner
发表于：IEEE会议论文集（1998）
中文翻译: http://www.aiqianji.com/blog/article/9

- 源码

比较热门的源码 有这些，现在一般pooling直接用maxpooling：

tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/lenet.py
tensorflow的输入 改成了28×28，因此少了一层卷积层，最后使用softmax输出

pytorch 源码 https://github.com/pytorch/examples/blob/master/mnist/main.py

caffe https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt

- pytorch code
- class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

- 知识点 ： 
反向传播算法  /  卷积  / 非线性激活函数  / 神经元结构  





### Day2 AlexNet

- 知识卡片  / 结构大图  / 论文网络结构图
 
- 简介:
2012年，Alex Krizhevsky用AlexNet 在当年的ImageNet图像分类竞赛中(ILSVRC 2012)，以top-5错误率15.3%拿下第一。 他的top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。它使用ReLU代替了传统的激活函数，而且网络针对多GPU训练进行了优化设计，用于提升速度。不过随着硬件发展，现在我们训练AlexNet都可以直接用简化后的代码来实现了。


- 特点：
1 ReLU函数作为激活函数
2 dropout:选择性地忽略训练中的单个神经元，避免模型的过拟合
3 max-pooling:避免平均池化（average pooling）的平均效应
4 利用双GPU NVIDIA GTX 580训练

其他： LRN层的优化，tf官方后来给出的代码，进行了修改，将初始化选择用xavier_initializer的方法，将LRN层移除了。

- 基本信息

论文： "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

作者：Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton.
发表于：2012

中文翻译:

中英文对照：

- 源码
有这些源码：
tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/alexnet.py

caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt

- pytorch code
- class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


- 知识点 ： 
RELU /  pooling  /dropout

### Day3 NIN

- 知识卡片  / 结构大图  / 论文网络结构图
 
- 简介:
2014年 ICLR 的paper，Network In Network(NIN)，改进了传统的CNN 网络，采用了少量参数进一步提高了 CIFAR-10、CIFAR-100 等数据集上的准确率：
1 Mlpconv Layer：Conv+MLP
2 Global Average Pooling


paper ：http://arxiv.org/abs/1312.4400

gitxiv: http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin

caffe: https://gist.github.com/mavenlin/d802a5849de39225bcc6

tensorflow： https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py


## 碎碎念
本来，只是打算个人继续抽空将AlphaTree 整理一下，然后但是有小伙伴加入了这个计划。
事情就变成了这样：

（*&*&&…………&￥%……￥……&**

譬如需要论文摘要 获取论文中的图或者公式， 就开发了论文解析翻译系统 还带论文更新和论文推荐…… 现在解析了几千篇了 @_@ 以后会有论文生成功能…… 要小声一点 不要被别人听到

譬如需要论文图放大，弄清晰一些，就加入了超分辨率 @_@


还有若干…… 甚至弄了个聊天机器人来陪着读论文是什么鬼 ^_^，（GAN生成的小姐姐 就是美 么么哒）下次记得帮我写点代码 谢谢 

据说还要加神秘功能……
哦 对了 还做了个网站…… 叫 AI千集  还是beta0.5版。  www.aiqianji.com  

此处经历了什么 可以脑补）


所以这个项目持续了很久的“准备”阶段。 额 本来 我只是想出个计划 帮大家刷刷题 刷刷论文，刷刷代码。

也许 这就是抛砖引玉吧。