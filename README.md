# 100-Days-Of-AI-Code
100 Days of Artificial Intelligence Coding。人工智能100天

之前推出了一个资源合集 AlphaTree计划，https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/
但是不少小伙伴反馈，很多时候资源太多，学起来无从下手 （无奈脸），往往都在收藏夹里吃灰。 于是打算推出一个 人工智能100天计划，带大家一起学习github上的项目和AI的基础知识。而且针对有的小伙伴英文不好的情况，贴心的我还提供了一些翻译链接，如果没有人翻译过，就提供论文机翻给到大家。

如果喜欢，就star一个吧。

## 为什么要做这样的一个AI100天计划

网上有不少机器学习的计划，也有一些roadmap，但是深度学习的不多，而且大多就是论文阅读列表。对于我们很多人来说，很难坚持下去。 
所以我从图像入手，将一些重要的论文 ，论文翻译，源代码，面试题 按发展时间，领域整合在一起。作为深度学习这部分学习的基础。

不同水平的小伙伴 可以选择自己合适的方法来学习
可以只看概述 刷题
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

#### 第四个月 GAN 生成对抗网络
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