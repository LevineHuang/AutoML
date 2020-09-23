模块

dataset

- BaseDataset(mx.gluon.data.Dataset)
- TabularDataset(pd.DataFrame)

task

- BaseTask
- TabularPrediction(BaseTask)
- ImageClassification(BaseTask)
- ObjectDetection(BaseTask)
- TextClassification(BaseTask)



## TabularPrediction(BaseTask)介绍

task.load()

task.fit()

task.predict

task.evaluate_predictions()

task.fit_summary()



#### task.fit()

初始化TabularDataset

从入参中获取trainer_type，否则默认为AutoTrainer。

eval_metric：在test data上进行评估的目标函数。

stopping_metric：用于early stop，以避免过拟合。默认与eval_metric一致，当eval_metric=‘roc_auc’时，stopping_metric的默认值为stopping_metric。

holdout_frac：training data中用于超参数调优的数据集的比例，根据training data的行数变化，	range from 0.2 at 2,500 rows to 0.01 at 250,000 rows。如果启动超参数自动调优，则该值翻倍，但不超过0.2。

num_bagging_folds：

stack_ensemble_levels：

enable_fit_continuation：训练的模型是否允许继续训练。如果允许，会将training data 和validation data保存至磁盘，以便将来使用。



```
Learner
DefaultLearner
AbstractLearner

AutoTrainer
AbstractTrainer
```

search_strategy：

- 'random' (random search)
- 'skopt' (SKopt Bayesian optimization)
- 'grid' (grid search)
-  'hyperband' (Hyperband)
- 'rl' (reinforcement learner)

feature_generator_type：AutoMLFeatureGenerator

trainer_type：AutoTrainer

scheduler_options

search_strategy及其相应的参数。

以上述参数初始化Learner。

```python
# Learner初始化
learner = Learner(path_context=output_directory, label=label, problem_type=problem_type, objective_func=eval_metric, stopping_metric=stopping_metric,
                          id_columns=id_columns, feature_generator=feature_generator, trainer_type=trainer_type,
                          label_count_threshold=label_count_threshold)

# 学习
learner.fit(X=train_data, X_test=tuning_data, scheduler_options=scheduler_options,
                    hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                    holdout_frac=holdout_frac, num_bagging_folds=num_bagging_folds, num_bagging_sets=num_bagging_sets, stack_ensemble_levels=stack_ensemble_levels,
                    hyperparameters=hyperparameters, time_limit=time_limits_orig, save_data=enable_fit_continuation, verbosity=verbosity)
```



### DefaultLearner(AbstractLearner)

**训练时间分配**

如果用户为指定训练时长time_limit，则默认为1e7。

从time_limit中减去general_data_processing数据处理时长，得到剩余时间time_limit_trainer用于Trainer。

```python
# 分配每个train level的时间
self.time_limit_per_level = (self.time_limit - (self.time_train_level_start - self.time_train_start)) / (level_end + 1 - level)
```



**数据处理**

针对所有模型的通用数据处理步骤。

1. 剔除训练集中标签列为NaN的行
2. 如果未指定任务类型，则根据标签列判断
3. 调整threshold, holdout_frac, num_bagging_folds的值
4. 如果是MULTICLASS问题，objective_func为’log_loss’，则对训练集进行augment_rare_classes(X)处理
5. 初始化Cleaner、LabelCleaner，对训练集特征列和标签列进行处理
6. 如果测试集不为空，则用5中相同的方法对特征列和标签列进行处理，若为空，则用初始化Learner指定的feature_generator对==训练集==继续进行处理，测试集返回None。



**初始化Trainer，启动训练**

1. 初始化

```python
trainer = self.trainer_type(
    path=self.model_context,
    problem_type=self.trainer_problem_type,
    objective_func=self.objective_func,
    stopping_metric=self.stopping_metric,
    num_classes=self.label_cleaner.num_classes,
    feature_types_metadata=self.feature_generator.feature_types_metadata,
    low_memory=True,
    kfolds=num_bagging_folds,
    n_repeats=num_bagging_sets,
    stack_ensemble_levels=stack_ensemble_levels,
    scheduler_options=scheduler_options,
    time_limit=time_limit_trainer,  # 时长约束
    save_data=save_data,
    verbosity=verbosity
)
```

2. 自动化训练

```python
trainer.train(X, y, X_test, y_test, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, holdout_frac=holdout_frac,
              hyperparameters=hyperparameters)
```

#### AutoTrainer(AbstractTrainer)

1. 获取预设置模型

   根据hyperparameters指定的基础模型范围及其超参数，初始化基础模型，返回基础模型列表

   ```
   models = self.get_models(hyperparameters, hyperparameter_tune=hyperparameter_tune)
   
   get_preset_models(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, stopping_metric=self.stopping_metric,
                                    num_classes=self.num_classes, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune)
   ```

2. 训练集、测试集合并拆分处理

   若为bagged_mode(kfolds >= 2时)，则将测试集合并至训练集，测试集置空；否则，进行训练集拆分：

   ```
   X_train, X_test, y_train, y_test = generate_train_test_split(X_train, y_train, problem_type=self.problem_type, test_size=holdout_frac)
   ```

3. 进行模型训练和集成

   ```python
   self.train_multi_and_ensemble(X_train, y_train, X_test, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
   
   ```



##### AbstractTrainer

1.将训练集、测试集保存至磁盘

2.train_multi_levels

**计算每个level的时间分配：time_limit_core、time_limit_aux，然后进行stack_new_level()。**

**将结果保存。**

```
self.train_multi_levels(X_train, y_train, X_test, y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level_start=0, level_end=self.stack_ensemble_levels)
```

- 计算每个level的时间分配：time_limit_core、time_limit_aux

  ```python 
  self.time_limit_per_level = (self.time_limit - (self.time_train_level_start - self.time_train_start)) / (level_end + 1 - level)
  
  # 时间分配
  time_limit_core = self.time_limit_per_level
  time_limit_aux = max(self.time_limit_per_level * 0.1, min(self.time_limit, 360))  # Allows aux to go over time_limit, but only by a small amount
  
  ```

- stack_new_level

  ```python
  self.stack_new_level_core(X=X, y=y, X_test=X_test, y_test=y_test, models=models, level=level, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, time_limit=time_limit_core)
  
  # 若为bagged_mode模式，则 
  self.stack_new_level_aux(X=X, y=y, level=level+1, time_limit=time_limit_aux)
  # 否则 
  self.stack_new_level_aux(X=X_test, y=y_test, fit=False, level=level+1, time_limit=time_limit_aux)
  ```

  1. stack_new_level_core
  2. stack_new_level_aux

- stack_new_level_core

  1. 若为bagged_mode，则将model进行StackerEnsembleModel

  2. 基于get_inputs_to_stacker()处理训练数据和测试数据

  3. 进行训练train_multi

     ```python
     self.train_multi(X_train=X_train_init, y_train=y, X_test=X_test, y_test=y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level=level, stack_name=stack_name, kfolds=kfolds, n_repeats=n_repeats, time_limit=time_limit)
     ```

- stack_new_level_aux

  1. 基于get_inputs_to_stacker()处理数据

  2. generate_weighted_ensemble()

     进行WeightedEnsembleModel实例化，然后将实例化结果传入train_multi进行训练

  3. 进行模型评估，model_performance()，将最好的模型作为weighted_ensemble_model返回

- train_multi

  ```python
  # stack_new_level_core
  self.train_multi(X_train=X_train_init, y_train=y, X_test=X_test, y_test=y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level=level, stack_name=stack_name, kfolds=kfolds, n_repeats=n_repeats, time_limit=time_limit)
  
  # stack_new_level_aux
  self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[weighted_ensemble_model], kfolds=kfolds, n_repeats=n_repeats, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, level=level, time_limit=time_limit)
  ```

  1. 若n_repeat_start == 0，则model_names_trained为初始化的train_multi_initial(),否则为传入的models。

     **train_multi_initial()**

     基于train_multi_fold()得到models_valid，追加到model_names_trained中返回。

     ```python
     models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=kfolds, n_repeats=n_repeats, n_repeat_start=0, level=level, time_limit=time_limit)
     ```

     **train_multi_fold()**

     基于self.train_single_full()训练上一步model_names_trained的每个模型，将训练好的模型列表返回。

     

  2. 如果：

     ```
     if (n_repeats > 1) and self.bagged_mode and (n_repeat_start < n_repeats):
         model_names_trained = self.train_multi_repeats(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=model_names_trained,
                                                        kfolds=kfolds, n_repeats=n_repeats, n_repeat_start=n_repeat_start, stack_name=stack_name, level=level, time_limit=time_limit)
     ```

     **train_multi_repeats()**

     基于self.train_single_full()对传入的models进行多次训练，将训练好的模型列表返回。

- train_single_full()

  对于每一个模型进行训练：

  1. 进行超参数调优模式

     如果模型为[BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]中的一种，则进行超参数调优

     ```
     hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X_train, y=y_train, k_fold=kfolds, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
     ```

     否则，从当前训练集中按比例拆分出训练集和测试集，然后进行超参数调优model.hyperparameter_tune()。

     

     **model.hyperparameter_tune()**

     ==重点==

     

     

  2. 不进行超参数调优模式

     ```
     model_names_trained = self.train_and_save(X_train, y_train, X_test, y_test, model, stack_name=stack_name, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, time_limit=time_limit)
     ```

     **train_and_save()**

     - 在分配的时间内进行单个模型的训练

     ```
     model = self.train_single(X_train, y_train, X_test, y_test, model, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, time_limit=time_limit)
     ```

     - 在测试集上评估所训练模型的效果
     - 将模型保存
     - 将model_names_trained返回



**保存trainer**

计算模型训练实际使用的时长



## 超参搜索

```
hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X_train, y=y_train, k_fold=kfolds, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)

```

1. 获取搜索空间

   根据任务类型获取对应的默认搜索空间，并剔除默认空间中用户定义的超参数空间，然后更新self.params。

2. 获取scheduler_func

   即为最初启动automl中的fit时定义的search_strategy，若启动了hyperparameter_tune，则该参数为必填项且为callable，默认为random。

   目前提供以下5种：

   schedulers = {
       'grid': FIFOScheduler,
       'random': FIFOScheduler,
       'skopt': FIFOScheduler,
       'hyperband': HyperbandScheduler,
       'rl': RLScheduler,
   }

3. 获取训练集和验证集

4. 注册一个model_trial，更新需要用到的参数

   ```
   model_trial.register_args(dataset_train_filename=dataset_train_filename,
       dataset_val_filename=dataset_val_filename, directory=directory, model=self, **params_copy)
   ```

5. 使用model_trial和scheduler_options来实例化scheduler_func

   - **FIFOScheduler(TaskScheduler)**
   - **HyperbandScheduler(FIFOScheduler)**
   - **RLScheduler(FIFOScheduler)**

6. 调度启动执行，scheduler.run()、scheduler.join_jobs()

7. 获取最优超参数搜索结果

   ```
   best_hp = scheduler.get_best_config() # best_hp only contains searchable stuff
   hpo_results = {'best_reward': scheduler.get_best_reward(),
                  'best_config': best_hp,
                  'total_time': time.time() - start_time,
                  'metadata': scheduler.metadata,
                  'training_history': scheduler.training_history,
                  'config_history': scheduler.config_history,
                  'reward_attr': scheduler._reward_attr,
                  'args': model_trial.args
                 }
   ```

8. 返回hpo_models, hpo_model_performances, hpo_results



### TaskScheduler(object)

1. 实例化RESOURCE_MANAGER和REMOTE_MANAGER，将用户配置的分布式节点IP列表dist_ip_addrs是厉害为node，加入到REMOTE_MANAGER，然后将REMOTE_MANAGER加入到RESOURCE_MANAGER，构成集群。

2.  Task、Resources、DistributedResource

   

### **FIFOScheduler(TaskScheduler)**

1. 获取超参数搜索任务的资源配置，若为定义则取默认值1cpu、0gpu，
2. 获取searcher方法，默认为Random searcher
3. 获取train_fn==哪儿定义的？==
4. 获取num_trials、time_out、max_reward、visualizer等
5. 启动每一次超参数搜索试验，如果超时或者从完成的任务重通过get_best_reward()获取最优reward，如果效果优于max_reward，则终止试验，否则继续下一次searcher推荐的超参数搜索试验
   - 判断config, extra_kwargs = self._promote_config()是否能够取到config，否则由searcher推荐config，config = self.searcher.get_config(**extra_kwargs)。
   - 初始化一个超参数搜索试验任务，task = Task(self.train_fn, {'args': self.args, 'config': config},
                         DistributedResource(**self.resource))，并添加到任务队列中，作为pending的待评估超参数组合。
   - 启动一个线程，执行该任务rp = threading.Thread(target=self._run_reporter, args=(task, job, reporter,
                                   self.searcher), daemon=False)
   - 从scheduled_tasks取任务执行，执行完追加到finished_tasks中

### HyperbandScheduler(FIFOScheduler)



### BaseSearcher(object)

### RandomSearcher(BaseSearcher)

1.初次搜索，从定义的搜索空间中通过 new_config = self.configspace.get_default_configuration().get_dictionary()获取默认的超参数组合，否则从搜索空间中随机采样超参数组合

### 





## 模型集成





## 图像分类

训练集被自动划分为训练集和验证集，根据模型在验证集上的效果表现搜索最优的超参数组合，最后用最优的超参数在整个数据集(训练集+验证集)上重新训练。