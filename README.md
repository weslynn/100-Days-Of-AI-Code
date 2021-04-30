# 100-Days-Of-AI-Code
100 Days of Artificial Intelligence Coding。人工智能100天

之前推出了一个资源合集 AlphaTree计划，https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/
但是不少小伙伴反馈，很多时候资源太多，学起来无从下手 （无奈脸），往往都在收藏夹里吃灰。 于是打算推出一个 人工智能100天计划，带大家一起学习github上的项目和AI的基础知识。而且针对有的小伙伴英文不好的情况，贴心的我还提供了一些翻译链接，如果没有人翻译过，就提供论文机翻给到大家。

如果喜欢，就star一个吧。


## 为什么要做这样的一个AI100天计划

网上有不少机器学习的计划，也有一些roadmap，但是深度学习的不多，而且大多就是论文阅读列表。对于我们很多人来说，很难坚持下去。
所以我从图像入手，将一些重要的论文 ，论文翻译，源代码， 按发展时间，领域整合在一起。作为深度学习这部分学习的基础。

如果大家有兴趣 再召唤其他系列 如 nlp 推荐，硬件机器人等

除了前两周最基础的网络结构外，根据不同的方向可以衍生出多种打卡模式。
每个同学也可以选择自己喜欢方式来进行。
初学者 可以只看知识卡片 或者后期加入刷题 对每个方向有个大概了解。 每天十五分钟。

进阶或者二刷的小伙伴 就可以详细看别人的打卡笔记 或者自己根据资料写打卡笔记
一篇文章和代码 全部吃透 需要点时间。一般一周之内吧。

会提供资料：
1 论文原文，机翻，或者 大家共享的翻译资料。
2 多种官方或者第三方源代码链接，项目提供的参考源码基于 python ＋pytorch 。
3 论文或者项目 的相关文章 （待开放 你们也可以自己推荐 ）


CC-BY-NC-SA 知识共享-署名-非商业性-相同方式共享


## 按照什么顺序来刷
本计划持续时间约4个月。


本计划分八个板块。以基础为主。想要对深度学习有个整个了解的，可以按顺序全刷　

－A　深度学习基础
－B　轻量级网络和大型网络
－C　物体检测
－D　物体分割
－E　人脸，文字的检测与识别
－F　肢体识别
－G　GAN基础
－H　GAN应用

想主要针对GAN的小伙伴 可以主刷 A G H E　，因为现在GAN在应用上，还是以人的生成较多，可以辅助F，换装系列　都要用到肢体相关的

图像分析，视频分析的小伙伴：可以试试　A C　D　E　

希望你们挖掘更多简洁的路径，也希望更多小伙伴加入到这个计划，跟进开发更新的方法和创新应用。

基础部分相对比较简单，每个网络大家尽可能动手去写一写。选择了一些相对轻量的数据库，如ｃｉｆａｒ１０，大部分小伙伴的机器还是跑得动的。

![ObjectClassification.png](http://aiqianji.oss-cn-shenzhen.aliyuncs.com/images/2021/04/17/36e14d6eff0f8deee26e5d828d0ba5ee.jpg)

### 第一个月 深度学习基础

- [Object Classification 物体分类]
- A：Week1 : 最基础的深度网络
  
  - [[LeNet]  掌握第一个商用的多层神经网络及其结构。 BP神经网络    
    知识点：反向传播算法  /  卷积  / 激活函数  / 神经元结构](https://aiqianji.com/blog/topic/10)
    中英文对照翻译：https://aiqianji.com/blog/article/9
    
  - [[AlexNet] 掌握8层的AlexNet的结构，细化多种激活函数
    知识点：RELU /  pooling  /dropout](https://aiqianji.com/blog/topic/11)
    中英文对照翻译：https://aiqianji.com/blog/article/20
    
  - [[NIN] 了解NIN的设计](https://aiqianji.com/blog/topic/12)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/12
    中英文对照翻译：https://aiqianji.com/blog/article/15

  - [[VGG]设计最简洁的VGG](https://aiqianji.com/blog/topic/13)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/13
    中英文对照翻译：https://aiqianji.com/blog/article/31

  - [[GoogLeNet]](https://aiqianji.com/blog/topic/15)
     知识卡片和网络介绍：https://aiqianji.com/blog/topic/15
    中英文对照翻译：https://aiqianji.com/blog/article/25
  - [Week1 总结](https://aiqianji.com/blog/topic/80)

---

- A：Week2 :出现ResNet之后，网络结构发生了各种变化
  
  - [ResNet](https://aiqianji.com/blog/topic/16)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/16
    中英文对照翻译：https://aiqianji.com/blog/article/34
  - [Inception V3](https://aiqianji.com/blog/topic/17)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/17
    中英文对照翻译： https://aiqianji.com/blog/article/30

  - [Inception-Resnet](https://aiqianji.com/blog/topic/74)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/74
    中英文对照翻译： https://aiqianji.com/blog/article/32

  - [DenseNet](https://aiqianji.com/blog/topic/75)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/75
    中英文对照翻译： https://aiqianji.com/blog/article/32

  - [DPN](https://aiqianji.com/blog/topic/85)
    知识卡片和网络介绍：https://aiqianji.com/blog/topic/85
    中英文对照翻译： https://aiqianji.com/blog/article/79
    
B ：Week3 轻量级模型 & 剪枝

- [MobileNet]
- [MobileNetV2]
- [SuffleNet]
- [SuffleNetV2]
- [SqueezeNet-SENet](https://aiqianji.com/blog/topic/72)
- 
- Week4： Resnet模型变种的发展 和自动学习
- [ResNeXt]
- [DPN]
- [WRN]
- [NasNet]
- [MNasNet]

![ObjectDetection&Seg.png](http://aiqianji.oss-cn-shenzhen.aliyuncs.com/images/2021/04/17/48bcde4633da580ee8ffda2f00670e58.jpg)

### 第二个月 物体检测和分割

- [Object Detection 物体检测](#object-detection-物体检测)
- C 1-2周 物体检测
  
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
- D 3-4周 物体分割
  
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

![face](http://aiqianji.oss-cn-shenzhen.aliyuncs.com/images/2021/04/17/592797dbd7b3c092f262440395794407.jpg)

- [ Detection & Recognition 基础检测与识别（人脸，文字）]
- E 1周 人脸与特征点检测
  
  - [MTCNN](#mtcnn-详解-detail-zhang-kaipeng-乔宇-qiao-yu--cuhk-mmlab--siat)
  - [Deep Face](#deep-face)
  - [FaceNet](#facenet-详解-detail)
  
  ![OCR.png](http://aiqianji.oss-cn-shenzhen.aliyuncs.com/images/2021/04/17/8c982ddeff85d86670d15e70935c7d33.jpg)
- [OCR](ocroptical-character-recognition-字符识别--str-scene-text-recognition-场景文字识别)
- E 2周 OCR相关
  
  - [CTPN](#ctpn-connectionist-text-proposal-network--详解-detail--zhi-tian--乔宇-qiao-yu--cuhk-mmlab--siat)
  - [TextBoxes](#textboxes--详解-detail-白翔-xiang-baimedia-and-communication-lab-hust)
  - [CRNN](#crnn-详解-detail-白翔-xiang-baimedia-and-communication-lab-hust)

![Pose.png](http://aiqianji.oss-cn-shenzhen.aliyuncs.com/images/2021/04/17/a02b4d1ad2bf2f4a376efba28d4e9aaa.jpg)

- F 2-3周 肢体检测

![GAN](https://github.com/weslynn/100-Days-Of-AI-Code/blob/master/map/Art%26Ganpic.png)

### 第四个月 GAN 生成对抗网络

![Art&Ganpic.png](http://aiqianji.oss-cn-shenzhen.aliyuncs.com/images/2021/04/17/4d4451ab73146d60054bbdc298bde0b9.jpg)

- [GAN 生成式对抗网络](#gan-生成式对抗网络)
- G 1-2周 GAN基础
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

H 3-4周 GAN应用


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
