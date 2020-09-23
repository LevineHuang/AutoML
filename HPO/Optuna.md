#### Eager search spaces

使用Python conditionals, loops和 syntax自动搜索最有超参数组合。

#### State-of-the-art algorithms

对大搜索空间进行高效搜索，对不良试验进行裁剪以快速得到搜索结果

#### 易并行

无需代码修改，即可通过多线程或多进行方式进行并行超参数搜索



如何预估终止条件

If you give neither `n_trials` nor `timeout` options, the optimization continues until it receives a termination signal such as Ctrl+C or SIGTERM. This is useful for use cases such as when it is hard to estimate the computational costs required to optimize your objective function.



