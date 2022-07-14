<h1> Hyper Parameter Optimization </h1>

<h3> Directory Structure </h3>

The data folder contains the data files as prepared from the data-prep. The sample config folder contains some sample configuration outputs that ray tune had compiled based on the specified experimental setting. Similarly the sample results folder contains some of the results that had been obtained from these experiments. These two folders present a specific setting and are not reproducible due to the later changes made in the execution. 

The src folder contains all the scripts that have been used. Each file contains the required comments and explanations. 

---
<h3> Experiment Scenarios </h3>

| Setting | Value | Meaning | Implementation in work | 
|-------|-------- | ------- | ------- |
| Learning Level | Single Task | use single task learning | Specify "STLAS" in '--task_name' argument |
|| Multi Task | use multi-task learning | Specify "MTLAS" in '--task_name' argument |
| Sampling Strategy | in topic | use data from all topics for training, validation and testing | default setting; no need to specific anything |
|| cross topic | randomly exclude a single topic from each dataset , for training and validation and use for testing | Specify "randomized" in 'task_name' argument |
| Data Source Selection | single dataset | Only use the specified data source for training (applicable only for single task learning) | Specify "only_" in '--task_name' argument |
|| Leave one out | Exclude the specified data source for training | Specify "LOO_" in '--task_name' argument |
|| All data sources | use all data sources | By default 'STLAS' and 'MTLAS' alone in --task_name' argument triggers this setting |
| include topic information | include topic | provide topic information as well to the encoder module | include "topic" in "task_name" argument |
|Aggregation Method (for inference)| mean| take mean of all logit values from the multiple seed runs to get the final logit value | Specify "mean" in 'task_name' argument |
|| variance | use the score value of the minimum variance regression unit | Specify "var" in 'task_name' argument |
|| weighted variance | use the score value of the minimum weighted variance regression unit | Specify "wt-var" in 'task_name' argument |

Data Source Codes used in the code:

- IBMArgQ : "gretz"
- IBMRank : "toledo"
- UKPConvArgRank : "ukp"
- SwanRank : "swanson"

Structure of the task name argument:
{a}\_{b}\_{c}\_{d}\_{e}

where "a" is required and takes value of "STLAS" or"MTLAS". "b" is optional and takes value of "only" or "LOO" or not mentioned (all included). "c" is optional and takes the value of any one of the data source codes (mentioned above). If "b" takes a value, then "c" is required to inform the system to include/exclude a data source from training. "d" is Optional and takes the value of "topic" or "source". "e" is required if doing inference, and takes the value of "mean", "var", or "wt-var".  

---
<h3> Execution </h3>

Here is a sample execution command to run a particular experiment setting.

<h5> For Hyperparameter Optimization </h5>

`python hpo_using_ray.py`

The config values for the different hyperparameters have been defined within the code. 

<h5> For training a model using a optimal config file </h5>

`python retraining.py`

The path to the config file is defined within the code script.

<h5> For inference (only for multi-task setting) </h5>

`python inference_using_MTL.py`

The aggregation of the logit values was done separately.

