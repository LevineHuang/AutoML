### AutoML-Zero：从零开始搜索机器学习算法

**梁辰 Google Brain研究员**



自动机器学习（AutoML）取得了很多的成功应用，但是目前的方法因为依赖于人工设计的搜索空间和模块而不够自动化。为了克服这些局限性，我们最近的AutoML-Zero工作探索了一个新的方向：从零开始用基本的数学操作去搜索机器学习算法，包括模型的结构和学习的策略。我们的实验展示了如何用进化算法重新发现包括反向传播在内的各种机器学习算法。



**内容大****纲：**

\1. Motivation: 如何克服目前AutoML方法不够自动化的局限性。

\2. AutoML-Zero framework：如何从基本的数学操作开始搜索机器学习算法。

\3. AutoML-Zero Experiments：用进化算法从零开始发现包括反向传播在内的各种机器学习算法。



![GIF for the experiment progress](assets/progress.gif)

我在想，有没有可能通过大量的AutoML-Zero实验，从学到的自动从中提取出