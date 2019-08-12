### <font color="red">人脸识别综述</font>[<font color="Red" size=3>(19 -Deep Face Recognition: A Survey)</font>](https://arxiv.org/abs/1804.06655)

人脸识别的四个发展阶段

<img src="face research/5.jpg" width="1000px">



##### <font color="blue" size=4>**概述**</font>

- 人脸识别各个模块
  - 人脸检测 - 定位存在人脸的位置
  - 人脸对齐 - 对齐人脸到正则坐标系
  - 人脸识别
    1. 活体检测
    2. 人脸识别 - **面部处理（处理姿态、亮度、表情、遮挡等）、特征提取、人脸比对**

<img src="face research/6.jpg" width="1000px">

- **面部处理 face processing**

  对**姿态（主要）**、亮度、表情、遮挡进行处理，可提升FR模型性能

  - one to many：从单个图像生成不同姿态的图像，使模型学习到不同的姿态
  - many to one：从多个不同姿态的图像中恢复正则坐标系视角下的图像，用于受限条件

- **特征提取 feature extraction**

  特征提取网络可分为backbone和assembled两类

  - backbone：用于提取特征的主干网络，如VGGNet，GoogleNet，SENet，ResNet。也有的将对齐和特征提取组装在一起形成联合网络
  - assembled：FR方法主要聚焦于训练这一组装网络，通过多输入或多任务以提升FR性能

- **损失函数 loss function**：通常intra-variations > inter-variations，softmax能学到可分特征，但是通常无法学习有足够区分度的特征，因此FR也聚焦于损失函数，即metric learning

  - Euclidean-distance-based loss：基于欧几里得距离增加组间距离、减小组内距离

  - angular/cosine-margin-based loss

  - softmax loss

- **面部匹配 face matching**

  对面部认证、面部识别任务，多数方法直接通过余弦距离或者L2距离直接计算两个特征图的相似性，再通过阈值对比threshold comparison或者最近邻NN判断是否为同一人

  此外，也可以通过Metric learning或者稀疏表示分类器sparse-representation-based classifier进行后处理，再进行特征匹配

- 面部识别的应用场景和研究方向

<img src="face research/7.jpg" width="500px">

##### <font color="blue" size=4>**面部处理（face processing）**</font>

- **One-to-many Augmentation**

  数据集的收集代价大，因此使用one-to-many augmentation对数据进行增光

  - Data augmentation

    photometric transformation

    geometric transformation：镜像、旋转、oversampling

  - 3D model

    对原2D人脸图像进行3D模型重建，再通过投影获得不同的姿态、表情等的2D人脸图像

  - CNN model

    通过CNN模型直接产生2D人脸图像

  - GAN model

    通过GAN生成对抗模型生成2D人脸图像

- **Many-to-one Normalization**

  many-to-one normalization通过多张人脸图像产生已对齐到正则坐标系的正脸图像，从而减少面部变化，使得面部对齐和特征比对更易实现

  - CNN、GAN等

##### <font color="blue" size=4>**损失函数（metric learning）**</font>

<img src="face research/8.jpg" width="1000px">

- **Euclidean-distance-based loss（上图绿色）**

  基于欧几里得距离损失是一种度量学习方法，它通过对输入图像提取特征将其嵌入欧几里得空间，然后减小组内距离、增大组间距离，包括contrastive loss，triplet loss，center loss和它们的变种

  - contrastive loss

    损失计算需要image pair，增加负例（两张图不同脸）距离，减少正例（同脸）距离。它考虑的是正例、负例之间的绝对距离，表达式为：

    $L=y_{ij}max(0, ||f(x_i)-f(x_j)||_2+\epsilon^+) + (1-y_{ij})max(0, \epsilon^--||f(x_i)-f(x_j)||_2))$

    其中：

    ​		$y_{ij}=1$表示$x_i,x_j$是正例pair，$y_{ij}=0$表示负例pair

    ​		$f(·)$表示特征嵌入函数

  - Triplet loss

    该损失计算需要triplet pair，三张图，分别为anchor, negative, positive。最小化anchor和positve间距离，同时最大化anchor和negative间距离，表达式为

    $L=\sum_i^N\{||f(x^a)-f(x^p)||_2^2-||f(x^a)-f(x^n)||_2^2+\alpha\}_+$

    注意，数据集中大多数的人脸之间都很容易区分，容易区分的triplet pair算出来的$L$很小，导致收敛缓慢，因此triplet pair选择的时候需要选择难以区分的人脸图像

  - Center loss

    该损失在原损失的基础上增加一个新的中心损失$L_C$，及每个样本与它的类别中心之间的距离，通过惩罚样本与距离间的距离来降低组内距离

    $L_C=\frac{1}{2}\sum_i^m||x_i-c_{y_i}||_2^2$，	其中$c_{y_i}$表示$x_i$所属的类的类别中心.

<img src="face research/8.jpg" width="1000px">

- **Angular/cosine-margin-based loss（黄色）**

  基于角度/余弦边缘损失，它使得FR网络学到的特征之间有更大的角度/余弦Margin

  - Softmax

    $L=-\frac{1}{N}\sum_ilog(\frac{e^{f_{y_i}}}{\sum_je^{f_j}})$

  ​		二分类的分类平面为：$(W_1-W_2)x+b_1-b_2=0$

  - L-Softmax

    令原始的Softmax loss中$f_j=||W_j||x_j||cos(\theta_j)$，同时增大$y_i$对应的项的权重可得到Large-margin softmax。该权重$m$引入了multiplicative angular/cosine margin

    <img src="face research/10.jpg" width="500px">
  
    其中
  
    <img src="face research/11.jpg" width="550px">
  
    二分类的分类平面为：$||x||(||W_1||cos(m\theta_1)-||W_2||cos(\theta_2))>0$
  
    <img src="face research/9.jpg" width="400px">
  
    L-softmax存在问题：收敛比较困难，$||W_1||,||W_2||$通常也不等
  
  - A-softmax (SphereFace)
  
    在L-softmax的基础上，将权重$L_2$正则化得到$||W||=1$，因此正则化后的权重落在一个超球体上
  
    <img src="face research/13.jpg" width="300px">
  
    二分类的分类超平面为：$||x||(cos(m\theta_1)-cos(\theta_2))=0$
  
    <img src="face research/12.jpg" width="500px">
  
  - CosFace / ArcFace
  
    与A-softmax相同思想，但CosFace/ArcFace引入的是additive angular/cosine margin
    
    CosFace：<img src="face research/14.jpg" width="400px">
    
    CosFace二分类的分类超平面为：$\widehat{x}(cos\theta_1-m-cos\theta_2)=0$
    
    ArcFace：<img src="face research/15.jpg" width="400px">
    
    ArcFace二分类的分类超平面：$\widehat{x}(cos(\theta_1+m)-cos\theta_2)=0$
    
    其中 $s=||x||$
  
  - 对比
  
    | 损失函数                  | 决策边界                                            |
    | ------------------------- | --------------------------------------------------- |
    | Softmax                   | $(W_1-W_2)x+b_1-b_2=0$                              |
    | L-softmax                 | $||x||(||W_1||cos(m\theta_1)-||W_2||cos\theta_2)>0$ |
    | A-softmax<br>(SphereFace) | $||x||(cos(m\theta_1)-cos(\theta_2))=0$             |
    | CosineFace                | $\widehat{x}(cos\theta_1-m-cos\theta_2)=0$          |
    | ArcFace                   | $\widehat{x}(cos(\theta_1+m)-cos\theta_2)=0$        |
  
    <img src="face research/16.jpg" width="500px">
  
    <img src="face research/8.jpg" width="1000px">
    
  - **Feature normalization（蓝色）**
  
    这类方法对softmax中的特征向量或者权重进行正则化，即：$\widehat{W}=\frac{W}{||W||},\widehat{x}=\alpha\frac{x}{||x||}$
  
    Feature normalization需要与其他损失函数共同使用，以达到最佳效果
    
    

##### <font color="blue" size=4>**网路结构 (Network architecture)**</font>

网络结构包括两方面：

主干网络（Backbone network）：一些通用的用于提取特征的网络

组装网络（Assembled network）：用于拼接在主干网络前/后的用于特定训练目标的网络

<font color="green">**Backbone Network**</font>

- Mainstream architectures

  主流的网络架构包括AlexNet，VGGNet，GoogleNet，ResNet，SENet等

<img src="face research/17.jpg" width="500px">

​			<img src="face research/18.jpg" width="1000px">

​		AlexNet：引入ReLU，dropout，data augmentation等，第一次在图像上有效使用Conv

​		VGGNet：提出重复用简单基础块堆叠；滤波器3x3减少铝箔其权重，增强表示能力

​		GoogleNet：1x1跨通道整合信息，同时用于升降维减少参数；并行结构由网络自行挑选最好的路径；多个出口计算不同位置损失，综合考虑不同层次的信息

​		ResNet：引入残差块，削弱层间联系，提高模型容忍度；信息跨层注入下游，恢复信息蒸馏过程中的信息丢失，减小表示平静的营销；残差块部分解决梯度消失

​		SENet：在上述网络中嵌入Squeeze-and-Excitation块，通过1x1块显式地构建通道间相互关系，自适应校准通道间的特征响应。Squeeze：全局平均池化得到1x1xC用于描述全局图像，使浅层也能获得全局感受野；Excitation：在FC-ReLU-FC-Sigmoid(类似门的作用)过程中得到各通道权重，然后rescale到WxHxC。从全局感受野和其它通道获得信息，SE块可自动根据每个通道的重要程度去提升有用的特征的权重，通过这个对原始特征进行重标定。

- **Special architectures**

  除了主流的最广泛使用的网络架构，还有一些特殊的模块和技巧，如max-feature-map activation，bilinear CNN，pairwise relational network等

- **Joint alignment-representation networks**

  这类模型将人脸检测、人脸对齐等融合到人脸识别的pipeline中进行端到端训练。比起分别训练各个部分的模型，这种端到端形式训练到的模型具有更强的鲁棒性

<font color="green">**Assembled Network**</font>

组装网络用于拼接在主干网前或后方，用于多输入或多任务的场景中

- **Multi-input networks**

  在one-to-many这类会生成不同部位、姿态的多个图像时，这些图片会输入到一个multi-input的组装子网络，一个子网络处理其中一张图片。然后将各个输出进行联结、组合等，再送往后续网络。

  如下图所示的多视点网络Multi-view Deep Network ([MvDN](http://openaccess.thecvf.com/content_cvpr_2016/html/Kan_Multi-View_Deep_Network_CVPR_2016_paper.html))进行cross-view recognition（对不同视角下的样本进行分类）

  <img src="face research/19.jpg" width="250px">

- **multi-task networks**

  在某些情景中，人脸识别是主要任务，若需要同时完成姿态估计、表情估计、人脸对齐、笑容检测、年龄估计等其余任务时，可以使用multi-task组装网。

  如下图Deep Residual EquivAriant Mapping ([DREAM](http://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Pose-Robust_Face_Recognition_CVPR_2018_paper.html))模块，用于特征层次的人脸对齐
  
  <img src="face research/20.jpg" width="650px">



##### <font color="blue" size=4>**面部匹配 (Face matching)**</font>

在测试阶段，通常使用余弦距离和L2距离来度量两个通过网络提取的深度特征$x_1,x_1$的相似性，再通过阈值比较threshold comparison进行面部验证face verification，通过最近邻分类器nearest neighbor classifier进行面部识别face identification

- Face verification

  1:1，如FaceID

- Face identification

  1:N



##### <font color="blue" size=4>**数据集 (Datasets)**</font>

<img src="face research/21.jpg" width="680px">

<img src="face research/22.jpg" width="400px">

- 数据集的Depth、Breadth

  - Depth

    不同人脸数较小，但每个人的图像数量很大。Depth大的数据集可以使模型能够更好的处理较大的组内变化intra-class variations，如光线、年龄、姿态。

    VGGface2（3.3M，9K）

  - Breadth

    不同人脸数较大，但每个人的图像数量较小。Breadth大的数据集可以使模型能够更好的处理更广范围的人群。

    MS-Celeb-1M（10M，100K）、MegaFace(Challenge 2，4.7M，670K)

- 数据集的Long tail distribution

  <img src="face research/23.jpg" width="200px">

- 数据集的data noise

  由于数据源和数据清洗策略的不同，各类数据集或多或少存在标签噪声label noise，这对模型的性能有较大的影响。

<img src="face research/24.jpg" width="400px">

- 数据集的data bias

  大多数数据集是从网上收集得来，因此主要为名人，并且大多大正式场合。因此这些数据集中的图像大多数是名人的微笑、带妆照片，年轻漂亮。这与从日常生活中获取的普通人的普通照片形成的数据集（Megaface）有很大的不同。

  另外，人口群体分布不均也会产生data bias，如人种、性别、年龄。通常女性、黑人、年轻群体更难识别。



##### <font color="blue" size=4>**Evaluation tasks and performance metrics**</font>

- **training protocols**

  - subject-dependent protocol：所有用于测试的图像中的ID已在训练集中存在，FR即一个特征可分的分类问题（不同人脸视为不同标签，为测试图像预测标签）。这一protocol仅适用于早期FR研究和小数据集。

  - subject-independent protocol：测试图像中的ID可能未在训练集中存在。这一protocol的关键是模型需要学得有区分度的深度特征表示

<img src="face research/25.jpg" width="1000px">

-  **Evaluation metric**

  - Face verification：性能评价指标通常为受试者操作特性曲线(ROC - Receiver operating characteric)，以及平均准确度(ACC)

    ROC：todo

    ACC：todo

  - Close-set face identification：rank-N，CMC (cumulative match characteristic)

    rank-N：todo

    CMC：todo

  - Open-set face identification：



##### <font color="blue" size=4>**FR应用场景**</font>

- **Cross-Factor Face Recognition**
  - Cross-Pose：正脸、侧脸，可使用one-to-many augmentation、many-to-one normalizations、multi-input networks、multi-task learning加以缓解
  - Cross-Age
  - Makeup

- **Heterogenous Face Recognition**

  - NIS-VIS FR：低光照环境中NIR (near-infrared spectrum 近红外光谱)成像好，因此识别NIR图像也是一大热门话题。但大多数数据集都是VIS (visual ligtht spectrum可见光光谱)图像。-- 迁移学习
  - Low-Resolution FR：聚焦提高低分辨率图像的FR性能

  - Phote-Sketch FR：聚焦人脸图像、素描间的转换。 -- 迁移学习、image2image学习

- **Multiple (or single) media Face Recognition**
  - Low-Shot FR：实际场景中，FR系统通常训练集样本很少(甚至单张)
  - Set/Template-based FR
  - Video FR：两个关键点，1. 各帧信息整合，2. 高模糊、高姿态变化、高遮挡

- **Face Recognition in Industry**
  - 3D FR
  - Partial FR：给定面部的任意子区域
  - Face Anti-attack：
  - FR for Mobile Device



### Architecture

##### [<font color="red">14cvpr - DeepFace</font>](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)

DeepFace: Closing the Gap to Human-level Performance in Face Verification

**Alignment pipeline**

<img src="face research/4.jpg" width="350px">

**DeepFace architecture**

<img src="face research/3.jpg" width="1000px">

L4~L6均为均布连接层：与卷积层类似，但是不同位置的卷积核参数不共享，因为各个特征（如嘴、鼻子、眼睛）只会出现在某些特定区域（嘴在中下方，鼻子在中间，眼睛在偏上。

[<font color="blue">15cvpr - FaceNet</font>](https://arxiv.org/abs/1503.03832)









### <font size=6>detect</font>

##### [<font color="red">19 - RetinaFace</font>](https://arxiv.org/abs/1905.00641)

RetinaFace: Single-stage Dense Face Localisation in the Wild

- **模型架构**

  <img src="face research/1.jpg" width="1000px"><img src="face research/2.jpg" width="1000px">

- **多任务损失函数**

  $L=L_{cls}(p_i,p_i^*)+\lambda_1p_i^*L_{box}(t_i,t_i^*)+\lambda_2p_i^*L_{pts}(l_i,l_i^*)+\lambda_3p_i^*L_{pixel},      \lambda_{1,2,3}=0.24, 0.1, 0.01$

  - 人脸分类损失 $L_{cls}(p_i,p_i^*)$

    $p_i$是预测第$i$个anchor中存在人脸的概率，$p_i^*=1/0$，$L_{cls}$为二分类的softmax loss

  - 人脸边框回归损失 $L_{box}(t_i,t_i^*)$

    $t_i=\{t_x,t_y,t_w,t_h\}$，$t_i^*$是预测, 实际的边界框，$L_{box}$是[fast rcnn](https://arxiv.org/abs/1504.08083)所定义的$R(t_i-t_i^*)$

  - 面部关键点回归损失$L_{pts}(l_i,l_i^*)$

    $l_i=\{l_{x1},l_{y1},...,l_{x5},l_{y5}\}$，$l_i^*$是正例的预测, 实际的关键点坐标

  - Dense回归损失$L_{pixel}$

    $L_{pixel}=\frac{1}{W*H}\sum_i^W\sum_j^H||R(D_{PST},P_{cam},P_{ill})_{(i,j)}-I_{(I,J)}^*||_1$
