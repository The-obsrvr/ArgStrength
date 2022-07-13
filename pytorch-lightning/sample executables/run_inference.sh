#!/bin/bash

## mtlas in-topic
#python inference.py --experiment=experiments/version_24-02-2022--23-26-05/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_24-02-2022--23-26-05/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_24-02-2022--23-26-05/ --aggregation_method=wt-var --gpus=3
#
##mtlas in-topic with topic
#python inference.py --experiment=experiments/version_25-02-2022--18-20-47/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_25-02-2022--18-20-47/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_25-02-2022--18-20-47/ --aggregation_method=wt-var --gpus=3
## mtlas cross-topic
#python inference.py --experiment=experiments/version_25-02-2022--08-53-19/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_25-02-2022--08-53-19/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_25-02-2022--08-53-19/ --aggregation_method=wt-var --gpus=3
##mtlas cross_topic with topic
#python inference.py --experiment=experiments/version_26-02-2022--02-46-30/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_26-02-2022--02-46-30/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_26-02-2022--02-46-30/ --aggregation_method=wt-var --gpus=3
#
# mtlas loo gretz in-topic
python inference.py --experiment=experiments/version_26-02-2022--10-26-06/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--10-26-06/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--10-26-06/ --aggregation_method=wt-var --gpus=3
# mtlas loo gretz cross-topic
python inference.py --experiment=experiments/version_26-02-2022--17-05-12/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--17-05-12/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--17-05-12/ --aggregation_method=wt-var --gpus=3
# mtlas loo gretz in-topic topic info
python inference.py --experiment=experiments/version_26-02-2022--23-03-14/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--23-03-14/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--23-03-14/ --aggregation_method=wt-var --gpus=3
# mtlas loo gretz cross-topic topic info
python inference.py --experiment=experiments/version_27-02-2022--02-55-02/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_27-02-2022--02-55-02/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_27-02-2022--02-55-02/ --aggregation_method=wt-var --gpus=3

## mtlas loo toledo in-topic
python inference.py --experiment=experiments/version_01-03-2022--23-30-39/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_01-03-2022--23-30-39/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_01-03-2022--23-30-39/ --aggregation_method=wt-var --gpus=3
## mtlas loo toledo cross-topic
python inference.py --experiment=experiments/version_26-02-2022--16-00-20/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--16-00-20/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_26-02-2022--16-00-20/ --aggregation_method=wt-var --gpus=3
## mtlas loo toledo in-topic topic info
python inference.py --experiment=experiments/version_03-03-2022--02-13-19/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_03-03-2022--02-13-19/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_03-03-2022--02-13-19/ --aggregation_method=wt-var --gpus=3
## mtlas loo toledo cross-topic topic info
python inference.py --experiment=experiments/version_27-02-2022--00-52-09/ --aggregation_method=mean --gpus=3
python inference.py --experiment=experiments/version_27-02-2022--00-52-09/ --aggregation_method=var --gpus=3
python inference.py --experiment=experiments/version_27-02-2022--00-52-09/ --aggregation_method=wt-var --gpus=3

## mtlas loo swanson in-topic
#python inference.py --experiment=experiments/version_27-02-2022--07-27-47/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--07-27-47/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--07-27-47/ --aggregation_method=wt-var --gpus=3
## mtlas loo swanson cross-topic
#python inference.py --experiment=experiments/version_27-02-2022--22-36-34/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--22-36-34/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--22-36-34/ --aggregation_method=wt-var --gpus=3
## mtlas loo swanson in-topic topic info
#python inference.py --experiment=experiments/version_28-02-2022--08-51-18/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_28-02-2022--08-51-18/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_28-02-2022--08-51-18/ --aggregation_method=wt-var --gpus=3
## mtlas loo swanson cross-topic topic info
#python inference.py --experiment=experiments/version_01-03-2022--13-31-18/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_01-03-2022--13-31-18/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_01-03-2022--13-31-18/ --aggregation_method=wt-var --gpus=3
#
## mtlas loo ukp in-topic
#python inference.py --experiment=experiments/version_26-02-2022--18-58-30/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_26-02-2022--18-58-30/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_26-02-2022--18-58-30/ --aggregation_method=wt-var --gpus=3
## mtlas loo ukp cross-topic
#python inference.py --experiment=experiments/version_27-02-2022--13-13-01/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--13-13-01/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--13-13-01/ --aggregation_method=wt-var --gpus=3
## mtlas loo ukp in-topic topic info
#python inference.py --experiment=experiments/version_27-02-2022--15-02-14/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--15-02-14/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--15-02-14/ --aggregation_method=wt-var --gpus=3
## mtlas loo ukp cross-topic topic info
#python inference.py --experiment=experiments/version_28-02-2022--12-00-46/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_28-02-2022--12-00-46/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_28-02-2022--12-00-46/ --aggregation_method=wt-var --gpus=3
#
## mtlas loo webis in-topic
#python inference.py --experiment=experiments/version_27-02-2022--08-01-29/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--08-01-29/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--08-01-29/ --aggregation_method=wt-var --gpus=3
## mtlas loo webis cross-topic
#python inference.py --experiment=experiments/version_27-02-2022--12-03-57/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--12-03-57/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--12-03-57/ --aggregation_method=wt-var --gpus=3
## mtlas loo webis in-topic topic info
#python inference.py --experiment=experiments/version_27-02-2022--13-07-10/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--13-07-10/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--13-07-10/ --aggregation_method=wt-var --gpus=3
## mtlas loo webis cross-topic topic info
#python inference.py --experiment=experiments/version_27-02-2022--17-47-43/ --aggregation_method=mean --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--17-47-43/ --aggregation_method=var --gpus=3
#python inference.py --experiment=experiments/version_27-02-2022--17-47-43/ --aggregation_method=wt-var --gpus=3
