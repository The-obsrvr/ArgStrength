<h1> Argument Quality </h1>


<h2> Setup: </h2>

`pip install -r requirements.txt`

<h2> Tasks </h2>

Keys (add these to the task name string to perform the respective objective/ the order does not matter):

1. STLAS      : Single Task Learning Argument Strength
2. MTLAS      : Multi Task Learning Argument Strength
3. topic      : add topic information to the tokenization Setup
4. source     : add source information to the tokenization Setup
5. only_{}    : only use the mentioned {} dataset
6. LOO_{}     : leave the mentioned {} dataset out.
7. randomized : define the train-validate-test sets by the topic distribution. 

**datasets**: "gretz", "toledo", "swanson", "ukp"

---

<h4> 1. Single Task Single Dataset </h4>

**Task Name string**: "STLAS_only_{}" or "STLAS_only_{}_randomized" or "STLAS_only\_{}_topic_randomized" ...

<h4> 2. Single Task Leave One Out Dataset </h4>

**Task Name string:** "STLAS_LOO_{}" or "STLAS_LOO_{}_randomized" ...

<h4> 3. Single Task All-in Dataset </h4>

**Task Name string**: "STLAS" or "STLAS_randomized" ...

<h4> 4. Multi Task Dataset </h4>

**Task Name string**: "MTLAS" or "MTLAS_randomized" ...

<h4> 5. Multi Task Leave One Out Dataset </h4>

**Task Name string**: "MTLAS_LOO_{}" or "MTLAS_LOO_{}_randomized" ...


Note:
For Task 4 and Task 5, we have further the options of different types of aggregation, for the logits obtained. We have the inference script defined for them.

<h2> Execution </h2>
