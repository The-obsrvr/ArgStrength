<h1> Hyper Parameter Optimization </h1>


<h3> Experiment Scenarios </h3>

| Setting | Value | Meaning | Implementation in work | 
|-------|-------- | ------- | ------- |
| Learning Level | Single Task | use single task learning | Specify "STLAS" in '--task_name' argument |
|| Multi Task | use multi-task learning | Specify "MTLAS" in '--task_name' argument |
| Sampling Strategy | in topic | use data from all topics for training, validation and testing | Specify "in-topic" in '--sampling_strategy' argument |
|| cross topic | randomly exclude a single topic from each dataset , for training and validation and use for testing | Specify "cross-topic" in '--sampling_strategy' argument |
|| balanced | select equal number of samples from each topic, keeping the size of each data set equal | Specify "balanced" in '--sampling_strategy' argument |
| Data Source Selection | single dataset | Only use the specified data source for training (applicable only for single task learning) | Specify "only_" in '--task_name' argument |
|| Leave one out | Exclude the specified data source for training | Specify "LOO_" in '--task_name' argument |
|| All data sources | use all data sources | By default 'STLAS' and 'MTLAS' in --task_name' argument triggers this setting |
| include topic information | include topic | provide topic information as well to the encoder module | use '--use_topic' argument |
|Aggregation Method (for inference)| mean| take mean of all logit values from the multiple seed runs to get the final logit value | Specify "mean" in '--aggregation_method' argument |
|| variance | use the score value of the minimum variance regression unit | Specify "var" in '--aggregation_method' argument |
|| weighted variance | use the score value of the minimum weighted variance regression unit | Specify "wt-var" in '--aggregation_method' argument |


<h3> Execution </h3>

Here is a sample execution command to run a particular experiment setting.

