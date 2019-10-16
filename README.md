# AutoML

## AutoML概述

### 人工智能定义

弱人工智能、强人工智能与超人工智能

人工智能三大主义

机器学习、深度学习、人工智能的关系

### 人工智能发展

第一、二、三、四段人工智能

下一代人工智能

人工智能--》应用智能

自学习的人工智能（<https://yq.aliyun.com/articles/88502>）

### AutoML基本概念

#### AutoM的提出背景

#### AutoML的定义

#### AutoM的研究动机

#### AutoML的意义和作用

### AutoML的研究领域

#### 按pipeline环节划分

##### AutoClean

##### AutoFE

##### AutoAugment

##### AutoNAS

##### HPO

##### Auto部署预测

##### Auto运维

##### AutoML Pipeline

从数据理解到模型部署整个机器学习全流程(pipeline)所有步骤的自动化。

#### 按应用领域划分

##### 传统机器学习

##### AutoST

##### AutoCV

图像分类

目标检测

##### AutoVoice

##### AutoNLP

##### AutoGraph

##### AutoGAN

## 数据准备

### 数据采集

### 数据清洗

### 数据标注

## 自动特征工程

### 自动化特征工程概述

### 自动化特征工程方法

### 自动化特征工程工具

### 自动化特征工程实践

## 自动模型选择/NAS

## 自动超参数优化

## AutoML模型的压缩与加速

## AutoML可视化

## AutoML在垂直领域的应用

## AutoML的生产化应用

可扩展性问题

资源调用

容错性、高可用

压缩和加速

## AutoML工具和框架

## AutoML平台和产品

## AutoML平台、系统架构设计

## AutoML助力AI中台

## AutoML前沿

## 附录

### AutoML相关期刊和会议

### AutoML研究学者

### AutoML学习资源

#### Automated Feature Engineering

##### Expand Reduce

- 2017 | AutoLearn — Automated Feature Generation and Selection | Ambika Kaul, et al. | ICDM | [`PDF`](https://ieeexplore.ieee.org/document/8215494/)
- 2017 | One button machine for automating feature engineering in relational databases | Hoang Thanh Lam, et al. | arXiv | [`PDF`](https://arxiv.org/pdf/1706.00327.pdf)
- 2016 | Automating Feature Engineering | Udayan Khurana, et al. | NIPS | [`PDF`](http://workshops.inf.ed.ac.uk/nips2016-ai4datasci/papers/NIPS2016-AI4DataSci_paper_13.pdf)
- 2016 | ExploreKit: Automatic Feature Generation and Selection | Gilad Katz, et al. | ICDM | [`PDF`](http://ieeexplore.ieee.org/document/7837936/)
- 2015 | Deep Feature Synthesis: Towards Automating Data Science Endeavors | James Max Kanter, Kalyan Veeramachaneni | DSAA | [`PDF`](http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf)

##### Hierarchical Organization of Transformations

- 2016 | Cognito: Automated Feature Engineering for Supervised Learning | Udayan Khurana, et al. | ICDMW | [`PDF`](http://ieeexplore.ieee.org/document/7836821/)

##### Meta Learning

- 2017 | Learning Feature Engineering for Classification | Fatemeh Nargesian, et al. | IJCAI | [`PDF`](https://www.ijcai.org/proceedings/2017/0352.pdf)

##### Reinforcement Learning

- 2017 | Feature Engineering for Predictive Modeling using Reinforcement Learning | Udayan Khurana, et al. | arXiv | [`PDF`](https://arxiv.org/pdf/1709.07150.pdf)
- 2010 | Feature Selection as a One-Player Game | Romaric Gaudel, Michele Sebag | ICML | [`PDF`](https://hal.archives-ouvertes.fr/inria-00484049/document)

#### NAS方法

##### **Random Search**

- [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569) | [2019/04]
- [Searching for Efficient Multi-Scale Architectures for Dense Image Prediction](http://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction) | [**NIPS 2018**]

##### **Bayesian Optimization**

- [Inductive Transfer for Neural Architecture Optimization](https://arxiv.org/abs/1903.03536) | [2019/03]

##### **Evolutionary Algorithm**

- [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) | [2019/04]
- [DetNAS: Neural Architecture Search on Object Detection](https://arxiv.org/abs/1903.10979) | [2019/03]
- [The Evolved Transformer](https://arxiv.org/abs/1901.11117) | [2019/01]
- [Designing neural networks through neuroevolution](https://www.nature.com/articles/s42256-018-0006-z) | [**Nature Machine Intelligence 2019**]
- [EAT-NAS: Elastic Architecture Transfer for Accelerating Large-scale Neural Architecture Search](https://arxiv.org/abs/1901.05884) | [2019/01]
- [Efficient Multi-objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081) | [**ICLR 2019**]

##### **Reinforcement Learning**

- [Template-Based Automatic Search of Compact Semantic Segmentation Architectures](https://arxiv.org/abs/1904.02365) | [2019/04]
- [Understanding Neural Architecture Search Techniques](https://arxiv.org/abs/1904.00438) | [2019/03]
- [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/abs/1901.07261) | [2019/01]
  - [falsr/FALSR](https://github.com/falsr/FALSR) | [Tensorflow]
- [Multi-Objective Reinforced Evolution in Mobile Neural Architecture Search](https://arxiv.org/abs/1901.01074) | [2019/01]
  - [moremnas/MoreMNAS](https://github.com/moremnas/MoreMNAS) | [Tensorflow]
- [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332) | [**ICLR 2019**]
  - [MIT-HAN-LAB/ProxylessNAS](https://github.com/MIT-HAN-LAB/ProxylessNAS) | [Pytorch, Tensorflow]
- [Transfer Learning with Neural AutoML](http://papers.nips.cc/paper/8056-transfer-learning-with-neural-automl) | [**NIPS 2018**]
- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) | [2018/07]
  - [wandering007/nasnet-pytorch](https://github.com/wandering007/nasnet-pytorch) | [Pytorch]
  - [tensorflow/models/research/slim/nets/nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) | [Tensorflow]
- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) | [2018/07]
  - [AnjieZheng/MnasNet-PyTorch](https://github.com/AnjieZheng/MnasNet-PyTorch) | [Pytorch]
- [Practical Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1708.05552) | [**CVPR 2018**]
- [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) | [**ICML 2018**]
  - [melodyguan/enas](https://github.com/melodyguan/enas) | [Tensorflow]
  - [carpedm20/ENAS-pytorch](https://github.com/carpedm20/ENAS-pytorch) | [Pytorch]
- [Efficient Architecture Search by Network Transformation](https://arxiv.org/abs/1707.04873) | [**AAAI 2018**]

##### **Gradient Based Methods**

- [Searching for A Robust Neural Architecture in Four GPU Hours](https://xuanyidong.com/publication/cvpr-2019-gradient-based-diff-sampler/) | [**CVPR 2019**]
  - [D-X-Y/GDAS](https://github.com/D-X-Y/GDAS) | [Pytorch]
- [ASAP: Architecture Search, Anneal and Prune](https://arxiv.org/abs/1904.04123) | [2019/04]
- [Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours](https://arxiv.org/abs/1904.02877#) | [2019/04]
  - [dstamoulis/single-path-nas](https://github.com/dstamoulis/single-path-nas) | [Tensorflow]
- [Automatic Convolutional Neural Architecture Search for Image Classification Under Different Scenes](https://ieeexplore.ieee.org/document/8676019) | [**IEEE Access 2019**]
- [sharpDARTS: Faster and More Accurate Differentiable Architecture Search](https://arxiv.org/abs/1903.09900) | [2019/03]
- [Learning Implicitly Recurrent CNNs Through Parameter Sharing](https://arxiv.org/abs/1902.09701) | [**ICLR 2019**]
  - [lolemacs/soft-sharing](https://github.com/lolemacs/soft-sharing) | [Pytorch]
- [Probabilistic Neural Architecture Search](https://arxiv.org/abs/1902.05116) | [2019/02]
- [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985) | [2019/01]
- [SNAS: Stochastic Neural Architecture Search](https://arxiv.org/abs/1812.09926) | [**ICLR 2019**]
- [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443) | [2018/12]
- [Neural Architecture Optimization](http://papers.nips.cc/paper/8007-neural-architecture-optimization) | [**NIPS 2018**]
  - [renqianluo/NAO](https://github.com/renqianluo/NAO) | [Tensorflow]
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | [2018/06]
  - [quark0/darts](https://github.com/quark0/darts) | [Pytorch]
  - [khanrc/pt.darts](https://github.com/khanrc/pt.darts) | [Pytorch]
  - [dragen1860/DARTS-PyTorch](https://github.com/dragen1860/DARTS-PyTorch) | [Pytorch]

##### **SMBO**

- [MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/abs/1903.06496) | [**CVPR 2019**]
- [DPP-Net: Device-aware Progressive Search for Pareto-optimal Neural Architectures](https://arxiv.org/abs/1806.08198) | [**ECCV 2018**]
- [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559) | [**ECCV 2018**]
  - [titu1994/progressive-neural-architecture-search](https://github.com/titu1994/progressive-neural-architecture-search) | [Keras, Tensorflow]
  - [chenxi116/PNASNet.pytorch](https://github.com/chenxi116/PNASNet.pytorch) | [Pytorch]

##### Local Search

- 2017 | Simple and Efficient Architecture Search for Convolutional Neural Networks | Thomoas Elsken, et al. | ICLR | [`PDF`](https://arxiv.org/pdf/1711.04528.pdf)

##### Meta Learning

- 2016 | Learning to Optimize | Ke Li, Jitendra Malik | arXiv | [`PDF`](https://arxiv.org/pdf/1606.01885.pdf)

##### Transfer Learning

- 2017 | Learning Transferable Architectures for Scalable Image Recognition | Barret Zoph, et al. | arXiv | [`PDF`](https://arxiv.org/abs/1707.07012)

##### Network Morphism

- 2018 | Efficient Neural Architecture Search with Network Morphism | Haifeng Jin, et al. | arXiv | [`PDF`](https://arxiv.org/abs/1806.10282)

##### Continuous Optimization

- 2018 | Neural Architecture Optimization | Renqian Luo, et al. | arXiv | [`PDF`](https://arxiv.org/abs/1808.07233)
- 2019 | DARTS: Differentiable Architecture Search | Hanxiao Liu, et al. | ICLR | [`PDF`](https://arxiv.org/abs/1806.09055)

**Hypernetwork:**

- [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/abs/1810.05749) | [**ICLR 2019**]

**Partial Order Pruning**

- Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search | [CVPR 2019]
  - [lixincn2015/Partial-Order-Pruning](https://github.com/lixincn2015/Partial-Order-Pruning) | [Caffe]

**Knowledge Distillation**

- [Improving Neural Architecture Search Image Classifiers via Ensemble Learning](https://arxiv.org/abs/1903.06236) | [2019/03]

#### NAS应用

##### **Image Classification**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](http://proceedings.mlr.press/v97/tan19a.html) | [**ICML 2019**]
  - [tensorflow/tpu/models/official/efficientnet/](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [Tensorflow]
  - [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) | [Pytorch]
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | [2019/05]
  - [kuan-wang/pytorch-mobilenet-v3](https://github.com/kuan-wang/pytorch-mobilenet-v3) | [Pytorch]
  - [leaderj1001/MobileNetV3-Pytorch](https://github.com/leaderj1001/MobileNetV3-Pytorch) | [Pytorch]

##### **Semantic Segmentation**

- [CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/abs/1811.08201) | [2019/04]
  - [wutianyiRosun/CGNet](https://github.com/wutianyiRosun/CGNet) | [Pytorch]
- [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431) | [2018/11]
  - [sacmehta/ESPNetv2](https://github.com/sacmehta/ESPNetv2) | [Pytorch]
- [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://sacmehta.github.io/ESPNet/) | [**ECCV 2018**]
  - [sacmehta/ESPNet](https://github.com/sacmehta/ESPNet/) | [Pytorch]
- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897) | [**ECCV 2018**]
  - [ooooverflow/BiSeNet](https://github.com/ooooverflow/BiSeNet) | [Pytorch]
  - [ycszen/TorchSeg](https://github.com/ycszen/TorchSeg) | [Pytorch]
- [ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) | [**T-ITS 2017**]
  - [Eromera/erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch) | [Pytorch]

##### **Object Detection**

- [ThunderNet: Towards Real-time Generic Object Detection](https://arxiv.org/abs/1903.11752) | [2019/03]
- [Pooling Pyramid Network for Object Detection](https://arxiv.org/abs/1807.03284) | [2018/09]
  - [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection/models) | [Tensorflow]
- [Tiny-DSOD: Lightweight Object Detection for Resource-Restricted Usages](https://arxiv.org/abs/1807.11013) | [**BMVC 2018**]
  - [lyxok1/Tiny-DSOD](https://github.com/lyxok1/Tiny-DSOD) | [Caffe]
- [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882) | [**NeurIPS 2018**]
  - [Robert-JunWang/Pelee](https://github.com/Robert-JunWang/Pelee) | [Caffe]
  - [Robert-JunWang/PeleeNet](https://github.com/Robert-JunWang/PeleeNet) | [Pytorch]
- [Receptive Field Block Net for Accurate and Fast Object Detection](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Songtao_Liu_Receptive_Field_Block_ECCV_2018_paper.pdf) | [**ECCV 2018**]
  - [ruinmessi/RFBNet](https://github.com/ruinmessi/RFBNet) | [Pytorch]
  - [ShuangXieIrene/ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch) | [Pytorch]
  - [lzx1413/PytorchSSD](https://github.com/lzx1413/PytorchSSD) | [Pytorch]
- [FSSD: Feature Fusion Single Shot Multibox Detector](https://arxiv.org/abs/1712.00960) | [2017/12]
  - [ShuangXieIrene/ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch) | [Pytorch]
  - [lzx1413/PytorchSSD](https://github.com/lzx1413/PytorchSSD) | [Pytorch]
  - [dlyldxwl/fssd.pytorch](https://github.com/dlyldxwl/fssd.pytorch) | [Pytorch]
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) | [**CVPR 2017**]
  - [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection/models) | [Tensorflow]

#### Neural Optimizatizer Search

- [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417) (Bello et al. 2017)

#### Hyper-Parameter Optimization

##### Black Box Optimization

###### Grid and Random Search

- 2017 | Design and analysis of experiments. | Montgomery | [`PDF`](https://support.sas.com/content/dam/SAS/support/en/books/design-and-analysis-of-experiments-by-douglas-montgomery/66584_excerpt.pdf)
- 2015 | Adaptive control processes: a guided tour. | Bellman | [`PDF`](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800080314)
- 2012 | Random search for hyper-parameter optimization. | Bergstra and Bengio | JMLR | [`PDF`](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)

###### Bayesian Optimization

- 2018 | Bohb: Robust and efficient hyperparameter optimization at scale. | Falkner et al. | JMLR | [`PDF`](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf)
- 2017 | On the state of the art of evaluation in neural language models. | Melis et al. | [`PDF`](https://arxiv.org/abs/1707.05589)
- 2015 | Automating model search for large scale machine learning. | Sparks et al. | ACM-SCC | [`PDF`](https://amplab.cs.berkeley.edu/wp-content/uploads/2015/07/163-sparks.pdf)
- 2015 | Scalable bayesian optimization using deep neural networks. | Snoek et al. | ICML | [`PDF`](https://arxiv.org/abs/1502.05700)
- 2014 | Bayesopt: A bayesian optimization library for nonlinear optimization, experimental design and bandits. | Martinez-Cantin | JMLR | [`PDF`](http://jmlr.org/papers/v15/martinezcantin14a.html)
- 2013 | Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. | Bergstra et al. | [`PDF`](http://proceedings.mlr.press/v28/bergstra13.pdf)
- 2013 | Towards an empirical foundation for assessing bayesian optimization of hyperparameters. | Eggensperger et al. | NIPS | [`PDF`](https://ml.informatik.uni-freiburg.de/papers/13-BayesOpt_EmpiricalFoundation.pdf)
- 2013 | Improving deep neural networks for LVCSR using rectified linear units and dropout. | Dahl et al. | IEEE-ICASSP | [`PDF`](http://ieeexplore.ieee.org/abstract/document/6639346/)
- 2012 | Practical bayesian optimization of machine learning algorithms. | Snoek et al. | NIPS | [`PDF`](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
- 2011 | Sequential model-based optimization for general algorithm configuration. | Hutter et al. | LION | [`PDF`](https://link.springer.com/chapter/10.1007/978-3-642-25566-3_40)
- 2011 | Algorithms for hyper-parameter optimization. | Bergstra et al. | NIPS | [`PDF`](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- 1998 | Efficient global optimization of expensive black-box functions. | Jones et al. | [`PDF`](http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf)
- 1978 | Adaptive control processes: a guided tour. | Mockus et al. | [`PDF`](https://www.researchgate.net/publication/248818761_The_application_of_Bayesian_methods_for_seeking_the_extremum)
- 1975 | Single-step Bayesian search method for an extremum of functions of a single variable. | Zhilinskas | [`PDF`](https://link.springer.com/article/10.1007/BF01069961)
- 1964 | A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. | Kushner | [`PDF`](https://fluidsengineering.asmedigitalcollection.asme.org/article.aspx?articleid=1431594)

###### Simulated Annealing

- 1983 | Optimization by simulated annealing. | Kirkpatrick et al. | Science | [`PDF`](https://science.sciencemag.org/content/220/4598/671)

###### Genetic Algorithms

- 1992 | Adaptation in natural and artificial systems: an introductory analysis with applications to biology, control, and artificial intelligence. | Holland et al. | [`PDF`](https://ieeexplore.ieee.org/book/6267401)

##### Multi-Fidelity Optimization

- 2019 | Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning. | Wu et al. | [`PDF`](https://arxiv.org/pdf/1903.04703.pdf)
- 2019 | Multi-Fidelity Automatic Hyper-Parameter Tuning via Transfer Series Expansion. | Hu et al. | [`PDF`](http://lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/aaai19_huyq.pdf)
- 2016 | Review of multi-fidelity models. | Fernandez-Godino | [`PDF`](https://www.arxiv.org/abs/1609.07196v2)
- 2012 | Provably convergent multifidelity optimization algorithm not requiring high-fidelity derivatives. | March and Willcox | AIAA | [`PDF`](https://arc.aiaa.org/doi/10.2514/1.J051125)

###### Modeling Learning Curve

- 2017 | Learning curve prediction with Bayesian neural networks. | Klein et al. | ICLR | [`PDF`](https://ml.informatik.uni-freiburg.de/papers/17-ICLR-LCNet.pdf)
- 2015 | Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. | Domhan et al. | IJCAI | [`PDF`](https://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf)
- 1998 | Efficient global optimization of expensive black-box functions. | Jones et al. | JGO | [`PDF`](http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf)

###### Bandit Based

- 2018 | Massively parallel hyperparameter tuning. | Li et al. | AISTATS | [`PDF`](https://arxiv.org/pdf/1810.05934.pdf)
- 2016 | Non-stochastic Best Arm Identification and Hyperparameter Optimization. | Jamieson and Talwalkar | AISTATS | [`PDF`](https://arxiv.org/abs/1502.07943)
- 2016 | Hyperband: A novel bandit-based approach to hyperparameter optimization. | Kirkpatrick et al. | JMLR | [`PDF`](http://www.jmlr.org/papers/volume18/16-558/16-558.pdf)

#### Model Compression & Acceleration

##### **Pruning**

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | [**ICLR 2019**]
  - [google-research/lottery-ticket-hypothesis](https://github.com/google-research/lottery-ticket-hypothesis) | [Tensorflow]
- [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270) | [**ICLR 2019**]
- [Slimmable Neural Networks](https://openreview.net/pdf?id=H1gMCsAqY7) | [**ICLR 2019**]
  - [JiahuiYu/slimmable_networks](https://github.com/JiahuiYu/slimmable_networks) | [Pytorch]
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494) | [**ECCV 2018**]
  - [AutoML for Model Compression (AMC): Trials and Tribulations](https://github.com/NervanaSystems/distiller/wiki/AutoML-for-Model-Compression-(AMC):-Trials-and-Tribulations) | [Pytorch]
- [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519) | [**ICCV 2017**]
  - [foolwood/pytorch-slimming](https://github.com/foolwood/pytorch-slimming) | [Pytorch]
- [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/abs/1707.06168) | [**ICCV 2017**]
  - [yihui-he/channel-pruning](https://github.com/yihui-he/channel-pruning) | [Caffe]
- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) | [**ICLR 2017**]
  - [jacobgil/pytorch-pruning](https://github.com/jacobgil/pytorch-pruning) | [Pytorch]
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) | [**ICLR 2017**]

##### **Quantization**

- [Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets](https://arxiv.org/abs/1903.05662) | [**ICLR 2019**]
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) | [**CVPR 2018**]
- [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342) | [2018/06]
- [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085) | [2018/05]
- [Post-training 4-bit quantization of convolution networks for rapid-deployment](https://arxiv.org/abs/1810.05723) | [**ICML 2018**]
- [WRPN: Wide Reduced-Precision Networks](https://arxiv.org/abs/1709.01134) | [**ICLR 2018**]
- [Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights](https://arxiv.org/abs/1702.03044) | [**ICLR 2017**]
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160) | [2016/06]
- [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432) | [2013/08]

##### **Knowledge Distillation**

- [Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://arxiv.org/abs/1711.05852) | [**ICLR 2018**]
- [Model compression via distillation and quantization](https://arxiv.org/abs/1802.05668) | [**ICLR 2018**]

##### **Acceleration**

- Fast Algorithms for Convolutional Neural Networks| [CVPR 2016]
  - [andravin/wincnn](https://github.com/andravin/wincnn) | [Python]

#### survey

#### book

#### 其它

### AutoML数据集和常用算法

### AutoML benchmark

### AutoML相关竞赛



### 















