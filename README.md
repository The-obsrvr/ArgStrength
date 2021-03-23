Link to Overleaf: https://www.overleaf.com/5561861524gdngbcwwzcqb 

Link to MLFlow Server: http://mlflow.dbs.ifi.lmu.de:5000/

## Task 1:

Task: Single Dataset models 
```
cd task1
python train.py --task train
```

## Task 2:

Task: Leave One Out Dataset Models

```
cd task2
python train.py --task train
```

## Task 3:

Task: All Datasets Models

```
cd task3
python task3_train.py --task train
```

## Task 4:

Task: Multi-Regression Learning Models
```
cd task4
python src/task4_run.py 
```
Default setting is for "equal" dataset weights. To change it to weighted, add the command line argument, `--dataset_loss_method weighted`



