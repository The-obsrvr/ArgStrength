<h1> Pytorch Lightning </h1>

<h3> Directory Structure </h3>

In the Data folder, the five data sets are available These have been pre-processed and structured uniformly as mentioned in the data-prep notebook found in the Data-prep directory in the home page. 

The src folder contains all the scripts that have been used for execution. Each script has been provided with useful comments and explanation to help improve understanding of each function and logic. The "datasets.py" contains information about the dataset class. This involves loading the data files as per the experiment setting, preparing the data, tokenizing it and compiling the training/validation/testing data sets to be loadable into the model. The "regressor.py" defines the model class as per the lightningmodule class. It includes defining the structure of the model (encoder unit and the regression unit), defining the training, validation, testing, performance calculation and inference steps based on the hyperparameter setting as set. The "training.py" script is used to define the training of the model as per the pytorch lightning setting. This includes the logging of parameters and metrics on MLflow, Early Stopping, checkpointing, ray tune based optimization, etc. Apart from this, the script can also be used to perform the task of testing for a given model. The "inference.py" script does the task of prediction for the given data file(s). The "utils.py" and "prepare_results.py" are auxilliary scripts providing some useful functions and supplementary tasks.

 Training specific arguments can be found listed in the training.py script (for training and testing) while inference specific arguments can be found listed in the inference.py script. Model specfic arguments can be found in the regressor.py script. For training, the hyperparameter arguments or the optimal model, obtained from the optimization experiments, have been pre-defined into the script.

 The results folder contains the results that had been obtained through the execution of the different experiments. The evaluation_results.csv  particular is fed to the prepare_results.py script to generate the final results in pandas and LaTeX format. 

 The sample executable folder contains some of the example execution commands used to run different experiment settings. 

---
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

Data Source Codes used in the code:

- IBMArgQ : "gretz"
- IBMRank : "toledo"
- UKPConvArgRank : "ukp"
- SwanRank : "swanson"
- Webis : "webis"

Structure of the task name argument:
{a}\_{b}\_{c}

where "a" is required and takes value of "STLAS" or"MTLAS". "b" is optional and takes value of "only" or "LOO" or not mentioned (all included). "c" is optional and takes the value of any one of the data source codes (mentioned above). If "b" takes a value, then "c" is required to inform the system to include/exclude a data source from training.

---
<h3> Example Execution </h3>

<h5> Training Example</h5>

<b> Single Task Learning for only SwanRank dataset with topic information and balanced sampling </b>

`python training.py --task_name=STLAS_only_swanson --gpus 0 --sampling_strategy balanced --train_batch_size=64`

<h5> Inference Example (only used for multi-task learning setting)</h5>

`python inference.py --experiment=experiments/version_26-02-2022--23-03-14/ --aggregation_method=wt-var --gpus=3`

The 'experiment' argument contains the path to the experiment folder containing the best checkpoint and the file containing the hyperparameter list.
