### 【顶会AAAI-2020】神经网络架构搜索(NAS)相关论文速览

1. #### Neural Graph Embedding for Neural Architecture Search. Wei Li, Shaogang Gong, Xiatian Zhu 

   现有的神经网络架构搜索（NAS）方法通常直接在离散或连续空间中搜索，而忽略了神经网络的图形拓扑知识。这会导致次优的搜索性能和效率，因为神经网络本质上是一个有向无环图（DAG）。 在这项工作中，我们通过引入神经图嵌入（neural graph embedding，NGE）的新颖思想解决了这一局限。具体来说，我们用一个神经DAG来表征神经网络的构建模块(例如，cell)，并通过一个图卷积网络来学习这个表征，从而传播和模拟网络结构的本质拓扑信息。这种方法获得了可集成现有不同NAS框架的通用神经网络表征。广泛的实验表明，NGE优于图像分类和语义分割任务上现有的SOTA方法。

2. #### Efficient Neural Architecture Search via Proximal Iterations. Quanming Yao, Ju Xu, Wei-Wei Tu, Zhanxing Zhu 

   神经架构搜索（NAS）能够搜索到比手工设计的更好的网络架构，因此吸引了很多研究关注。最近，差分搜索方法成为NAS方面的最新技术，它可以在几天时间之内获得高性能的网络架构。但由于supernet的构建，这类方法计算成本巨大、性能表现不佳。在本文中，我们提出了一种基于近似迭代的高效NAS方法（NASP）。与以前的研究工作不同，NASP将搜索过程重新定义为具有离散约束条件的优化问题和模型复杂度的正则化。由于新目标函数难以求解，受优化中近似迭代的启发，我们进一步提出了一种高效算法。通过这种方式，NASP不仅比现有的差分搜索方法更快，也可以找到更好的网络架构并平衡模型的复杂性。在各种任务上的实验表明，NASP可以以超过当前SOTA方法10倍以上的加速获得高性能的网络架构。

3. #### Binarized Neural Architecture Search. Hanlin Chen, Li'an Zhuo, Baochang Zhang, Xiawu Zheng, Jianzhuang Liu, David Doermann, Rongrong Ji 

   通过为各种任务自动设计最佳的神经网络架构，神经体系结构搜索（NAS）可对计算机视觉产生重大影响。具有二元卷积搜索空间的二值化神经体系结构搜索（BNAS）可以搜索到极度压缩的模型。但该领域仍未得到充分的探索研究。 由于优化需求导致学习效率低下以及巨大的搜索空间，BNAS比NAS更具挑战性。为了解决这些问题，我们将通道采样和操作空间缩减引入到了差分NAS方法中，以显着降低成本开销。这是通过基于性能的策略来丢弃一些具有较少潜力的操作来实现的。我们使用了两种二值化神经网络的优化方法来验证BNAS的有效性。大量实验表明，我们提出的BNAS在CIFAR和ImageNet数据集上均具有与NAS相当的性能表现。在CIFAR-10数据集上，准确率达到96.53％（差分NAS的准确率为97.22％），模型压缩效果显著，搜索速度比最新的PCDARTS快40％。

4. #### AutoShrink: A Topology-Aware NAS for Discovering Efficient Neural Architecture. Tunhou Zhang, Hsin-Pai Cheng, Zhenwen Li, Feng Yan, Chengyu Huang, Hai Li, Yiran Chen 

   在移动设备和边缘设备上部署深度神经网络时，资源是重要的限制。现有研究工作通常采用基于cell的搜索方法，限制了学习单元中网络模式的灵活性。另外，由于基于cell和基于node方法的现有研究中拓扑不可知的性质，搜索过程都很耗时，发现的网络结构的性能可能不是最佳的。为了解决这些问题，我们提出了AutoShrink，这是一种拓扑感知的神经体系结构搜索方法，用于搜索神网络结构的有效构建块。我们的方法基于node，因此可以在拓扑搜索空间内学习单元结构中灵活的网络模式。有向无环图（DAG）用于抽象DNN架构并通过边缘收缩(edge shrinking)方式逐步优化单元结构。由于随着边缘逐渐缩小，搜索空间本质上也会缩小，AutoShrink能够在更短的搜索时间探索更多灵活的搜索空间。我们通过设计ShrinkCNN和ShrinkRNN模型，在图像分类和语言任务上评估AutoShrink的作用。 ShrinkCNN在mageNet-1K数据集上与当前SOTA模型精度相当，但可减少48％的参数并节省34％I的Multiply-Accumulates (MACs) 计算。具体来说，在1.5 GPU小时内训练得到的ShrinkCNN和ShrinkRNN模型分别比SOTA CNN和RNN模型快7.2倍和6.7倍的时间。

5. #### InstaNAS: Instance-Aware Neural Architecture Search. An-Chieh Cheng, Chieh Hubert Lin, Da-Cheng Juan, Wei Wei, Min Sun 

   常规的神经网络架构搜索（NAS）旨在找到一个可以实现最佳性能的架构，通常可以优化与任务相关的学习目标，例如准确性。但是，单个网络架构可能对于具有高多样性的整个数据集而言，代表性不足。直观来看，选择精通特定领域特征的领域专家架构(domain expert architecture)可以进一步使与架构相关的目标受益，例如时延。在本文中，我们提出了InstaNAS（一种实例感知NAS框架），该框架采用了一个控制器，训练后用来搜索“架构的分布”而不是单个网络架构；这允许模型对于困难样本使用复杂的架构，对于简单样本的使用浅层的架构。在推断阶段，可以通过自定义推理成本，控制器为每个新的输入样本分配一个实现高精度的领域专家架构(domain expert architecture)。在一个受MobileNetV2启发的搜索空间中，实验显示，InstaNAS可以在一系列数据集上获得与MobileNetV2相当的准确性，并实现高达48.8％的延迟减少。

7. #### Posterior-Guided Neural Architecture Search. Yizhou Zhou, Xiaoyan Sun, Chong Luo, Zheng-Jun Zha, Wenjun Zeng 

   神经网络架构搜索（NAS）的出现极大地推动了网络设计的研究。最近研究提出的方法，例如基于梯度的方法或one-shot方法，大大提高了NAS的效率。在本文中，我们从贝叶斯角度阐述了NAS问题。我们建议在成对的网络架构和权重上显式估计联合后验分布。相应地，提出了一种混合网络表示形式，它使我们能够利用“Variational Dropout”，以便使后验分布的近似变得完全基于梯度(gradient-based)且高效。然后，提出了一种后验指导的采样方法来对候选架构进行采样并直接进行评估。作为一种贝叶斯方法，我们提出的后验指导的NAS（posterior-guided NAS , PGNAS）避免了调整多个超参数，并能够在后验概率空间中进行非常高效的架构采样。有趣的是，它还可以使我们更深入地了解one-shot NAS中使用的权重分配，并自然地减轻了由于权重共享导致的架构采样与权重共享之间的不匹配问题。我们在图像分类任务上验证了PGNAS方法。 在Cifar-10，Cifar-100和ImageNet数据集上的结果表明，在众多NAS方法中，PGNAS在搜索精度和搜索速度之间取得了很好的权衡。例如，经过11 GPU days的训练搜索到了一个非常有竞争力的网络架构，在Cifar10和Cifar100上的测试错误率分别为1.98％和14.28％。

8. #### Neural Architecture Search Using Deep Neural Networks and Monte Carlo Tree Search. Linnan Wang, Yiyang Zhao, Yuu Jinnai, Yuandong Tian, Rodrigo Fonseca

   神经网络架构搜索（NAS）在自动化神经网络设计方面已取得了巨大的成功，但是当前NAS方法背后的计算量过大，需要进一步研究以提高采样效率和网络评估成本，从而在较短的时间内获得更好的结果。在本文中，我们提出了一种新颖的基于NAS代理的可扩展蒙特卡罗树搜索（MCTS）方法，即AlphaX，以解决这两个方面的问题。 AlphaX通过在状态层适应性地平衡探索和利用，并通过元深度神经网络（Meta-Deep Neural Network，DNN）预测神经网络的准确性，从而使搜索偏向更有希望的区域，从而提高了搜索效率。为了分摊神经网络的评估成本，AlphaX通过分布式设计加快了MCTS的部署，并通过迁移学习（以MCTS中的树结构为指导）减少了神经网络评估时的迭代轮数。在1000个样本中通过12 GPU days训练，AlphaX发现了一种架构，在CIFAR-10和ImageNet上的top-1精度分别达到97.84％和75.5％，在准确性和采样效率上均超过了SOTA NAS方法。特别是，我们还在大型NAS数据集NASBench-101上评估了AlphaX。在寻找全局最优值方面，AlphaX的采样效率比随机搜索（Random Search）和正则化进化（Regularized Evolution）方法分别高3倍和2.8倍。该方法搜索到的网络结构改善了从神经网络风格迁移到图题生成、目标检测等各种视觉应用。

8. #### TextNAS: A Neural Architecture Search Space Tailored for Text Representation. Yujing Wang, Yaming Yang, Yiren Chen, Jing Bai, Ce Zhang, Guinan Su, Xiaoyu Kou, Yunhai Tong, Mao Yang, Lidong Zhou 

   学习文本表征对于文本分类和其它与语言相关的任务至关重要。 文献中有各种各样的文字表征网络，如何找到最优表征网络是一个十分重要的问题。 最近，新兴神经网络结构搜索（NAS）技术已显示出解决该问题的良好潜力。 尽管如此，大多数现有的NAS研究集中在搜索算法上，在搜索空间上的研究较少。我们认为，NAS要在不同应用上取得成功，搜索空间也是重要的人类先验知识。 因此，我们提出一种为文本表征量身定制的新颖搜索空间。 通过自动搜索后，发现的神经网络结构的性能在文本分类和自然语言推理任务的各种公共数据集上优于SOTA模型。 此外，自动搜索发现的一些网络设计原则也非常符合人类的直觉。

9. #### Towards Oracle Knowledge Distillation with Neural Architecture Search. Minsoo Kang, Jonghwan Mun, Bohyung Han 

   本文提出了一种新颖的知识蒸馏框架，该框架能够从集成教师网络中学习强大而高效的学生模型。我们的方法解决了教师和学生之间固有的模型能力问题（model capacity issue），旨在通过减少它们的能力差距来最大程度地提高教师模型在蒸馏过程中带来的益处。具体来说，我们采用了神经网络架构搜索技术来增强有用的结构和操作，使搜索到的网络适合于向学生模型进行知识提炼，并且不会通过固定网络容量而牺牲其性能。本研究还引入了Oracle知识蒸馏损失，以使用基于集成的教师模型促进模型搜索和蒸馏，从而学习一个学生网络以模仿教师网络的Oracle性能。我们使用了各种网络在图像分类数据集（CIFAR-100和TinyImageNet）上进行了广泛的实验。结果表明，寻找新的学生模型在准确性和内存大小上都是有效的，而且由于采用了Oracle知识提炼的神经网络架构搜索，搜索到的模型通常优于其教师模型。

10. #### Ultrafast Photorealistic Style Transfer via Neural Architecture Search. Jie An, Haoyi Xiong, Jun Huan, Jiebo Luo 

    逼真图像风格迁移的关键挑战是，算法能够忠实地将参考照片的风格迁移到内容照片中，而生成的图像应看起来像由相机拍摄的图像。尽管目前已经有几种逼真风格迁移算法，但是它们需要依赖后处理和/或预处理来使生成的图像看起来逼真。如果我们禁用附加处理，则这些算法将无法在细节保留和照片拟真的方面产生合理的照片风格拟真。在这项工作中，我们提出了针对这些问题的有效解决方案。我们的方法包括用于构建照片风格迁移网络的构建步骤（C步骤）和用于加速的裁剪步骤（P步骤）。在C步骤中，我们基于经过精心设计的预分析，提出了一个名为PhotoNet的密集自动编码器。 PhotoNet集成了特征聚合模块（BFA）和实例标准化的skip links（INSL）。为了产生忠实的样式，我们在解码器和INSL中引入了多个样式迁移模块。在效率和有效性方面，PhotoNet明显优于现有算法。在P步中，我们采用了神经网络架构搜索算法来加速PhotoNet。我们以师生学习的方式提出了一种自动网络修剪框架，以实现逼真的样式。通过搜索得到的名为PhotoNAS的网络结构可在PhotoNet上实现显着的加速，同时几乎完好无损的保持了样式效果。在图像和视频风格迁移上的广泛实验结果表明，与现有的最新方法相比，该方法可产生令人满意的结果，同时实现20-30倍的加速。值得注意的是，所提出的算法无需任何预处理或后处理即可实现更好的性能。

11. #### SM-NAS: Structural-to-Modular Neural Architecture Search for Object Detection. Lewei Yao, Hang Xu, Wei Zhang, Xiaodan Liang, Zhenguo Li 

    现有目标检测方法由于backbone、feature fusion neck、RPN、和RCNN head等各种模块而变得复杂，其中每个模块可能具有不同的设计和结构。在结构组合以及多个模块的模块化选择中如何进行计算成本和准确性之间的权衡呢？神经架构搜索（NAS）在寻找最优解决方案方面显示出巨大潜力。现有的用于目标检测的NAS仅专注于搜索单个模块（例如backbone和feature fusion neck）的更好设计，而忽略了整个系统的平衡。在本文中，我们提出了一种称为结构到模块NAS（Structural-to-Modular NAS，SM-NAS）的由粗到精的两步搜索策略，用于为目标检测任务搜索GPU友好的网络设计，既有效地组合了模块，又找到了更好的模块级网络结构。具体而言，结构级（Structural-level ）搜索阶段的首要目标是找到不同模块的有效组合；然后，模块级（Modular-level）搜索阶段会扩展每个特定的模块，并将Pareto front 推向一个更快的特定任务的网络。我们考虑了一种多目标搜索，其中搜索空间涵盖了许多流行检测方法的架构设计。通过探索一种从头开始的快速训练的策略，我们可以直接搜索目标检测backbone而无需预先训练的模型或任何代理任务。最终的网络架构在推理时间和准确性两方面都领先于最先进的目标检测系统，并在多个检测数据集上证明了其有效性。与FPN相比，推理时间减少了一半，mAP提升了1％，取得了46%的mAP，推理时间与MaskRCNN的相似。

12. #### An ADMM Based Framework for AutoML Pipeline Configuration. Sijia Liu, Parikshit Ram, Deepak Vijaykeerthy, Djallel Bouneffouf, Gregory Bramble, Horst Samulowitz, Dakuo Wang, Andrew Conn, Alexander Gray 

13. #### M-NAS: Meta Neural Architecture Search. Jiaxing Wang, Jiaxiang Wu, Haoli Bai, Jian Cheng