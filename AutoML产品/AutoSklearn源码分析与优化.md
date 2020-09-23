## 源码分析

1. 元学习所有数据集

## 使用方法

### 度量指标切换方法

```python
# 默认指标
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.992908
  Number of target algorithm runs: 101
  Number of successful target algorithm runs: 101
  Number of crashed target algorithm runs: 0
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

# autosklearn.metrics.f1
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: f1
  Best validation score: 0.994413
  Number of target algorithm runs: 86
  Number of successful target algorithm runs: 84
  Number of crashed target algorithm runs: 1
  Number of target algorithms that exceeded the time limit: 1
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.951048951048951

```

### 基础算法搜索空间定义

**include_estimators**list, optional (None)

If None, all possible estimators are used. Otherwise specifies set of estimators to use.

**exclude_estimators**list, optional (None)

If None, all possible estimators are used. Otherwise specifies set of estimators not to use. Incompatible with include_estimators.

## 优化试验

### 二分类问题

基础算法搜索空间优化，不同搜索空间上的多次重复试验。

baseline：默认的所有基础算法

exp1.1: AutoGluon中的6个基础算法

```sh
# baseline
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.992908
  Number of target algorithm runs: 111
  Number of successful target algorithm runs: 109
  Number of crashed target algorithm runs: 2
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.951048951048951

# 1.1
# ['adaboost', 'extra_trees']
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.985816
  Number of target algorithm runs: 119
  Number of successful target algorithm runs: 112
  Number of crashed target algorithm runs: 7
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.951048951048951

# 1.2 ['adaboost', 'extra_trees', 'random_forest', 'k_nearest_neighbors']
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.992908
  Number of target algorithm runs: 91
  Number of successful target algorithm runs: 90
  Number of crashed target algorithm runs: 0
  Number of target algorithms that exceeded the time limit: 1
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.958041958041958

# 1.3 ['adaboost', 'extra_trees', 'random_forest', 'k_nearest_neighbors', 'sdg']
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.985816
  Number of target algorithm runs: 143
  Number of successful target algorithm runs: 142
  Number of crashed target algorithm runs: 1
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.951048951048951

auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.985816
  Number of target algorithm runs: 128
  Number of successful target algorithm runs: 128
  Number of crashed target algorithm runs: 0
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.958041958041958

auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.985816
  Number of target algorithm runs: 137
  Number of successful target algorithm runs: 133
  Number of crashed target algorithm runs: 4
  Number of target algorithms that exceeded the time limit: 0
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.951048951048951

```





refit

```sh
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.936364
  Number of target algorithm runs: 195
  Number of successful target algorithm runs: 184
  Number of crashed target algorithm runs: 9
  Number of target algorithms that exceeded the time limit: 2
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.932

# refit
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.937121
  Number of target algorithm runs: 329
  Number of successful target algorithm runs: 312
  Number of crashed target algorithm runs: 13
  Number of target algorithms that exceeded the time limit: 4
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score 0.93

# train1
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.937879
  Number of target algorithm runs: 459
  Number of successful target algorithm runs: 437
  Number of crashed target algorithm runs: 15
  Number of target algorithms that exceeded the time limit: 7
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.932
F1 score:  0.22727272727272727

# train
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.937879
  Number of target algorithm runs: 577
  Number of successful target algorithm runs: 549
  Number of crashed target algorithm runs: 17
  Number of target algorithms that exceeded the time limit: 11
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.934
F1 score:  0.33999999999999997

#############
# train fron scratch on train2
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.934848
  Number of target algorithm runs: 184
  Number of successful target algorithm runs: 176
  Number of crashed target algorithm runs: 5
  Number of target algorithms that exceeded the time limit: 3
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.932
F1 score:  0.3461538461538461

# train on train2 and refit on train1
# initial_configurations_via_metalearning=0,

auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.934848
  Number of target algorithm runs: 346
  Number of successful target algorithm runs: 324
  Number of crashed target algorithm runs: 15
  Number of target algorithms that exceeded the time limit: 7
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.922
F1 score:  0.32758620689655177

# train on train2 and refit on train1
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.934848
  Number of target algorithm runs: 512
  Number of successful target algorithm runs: 482
  Number of crashed target algorithm runs: 20
  Number of target algorithms that exceeded the time limit: 10
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.932
F1 score:  0.2608695652173913

# # train on train2 and refit on train
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.935606
  Number of target algorithm runs: 618
  Number of successful target algorithm runs: 577
  Number of crashed target algorithm runs: 23
  Number of target algorithms that exceeded the time limit: 18
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.928
F1 score:  0.4

# # train on train2 and refit on train
auto-sklearn results:
  Dataset name: breast_cancer
  Metric: accuracy
  Best validation score: 0.935606
  Number of target algorithm runs: 740
  Number of successful target algorithm runs: 692
  Number of crashed target algorithm runs: 26
  Number of target algorithms that exceeded the time limit: 22
  Number of target algorithms that exceeded the memory limit: 0

Accuracy score:  0.929
F1 score:  0.36036036036036034
```

