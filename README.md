# Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections

Ruiqi Zhong, Kristy Lee<sup>*</sup>, Zheng Zhang<sup>*</sup>, Dan Klein

EMNLP 2021 Findings, https://arxiv.org/abs/2104.04670

## Data

Each datapoint is represented as a dictionary.

```{"q": [label description], "c": [text input], "a": [0 or 1]}```, 

where "q" stands for question, which contains label information, "c" stands for context, which contains the input text, "a" stands for answer, which is either 1 (Yes) or 0 (No). 

```training_dicts/``` contains all the datasets for training, and each of the .pkl file is a list of datapoints. 
```testing_dicts/``` contains all the datasets for evaluation, and each of the .pkl file is a map from (label, label descriptions) to a list of datapoints.

Datasets that have the same group number in front of their filenames are considered similar. 
Notice that,  for each dataset, there might be overlapping datapoints between the training and testing split, but it is okay since we never train and test on the same dataset.


## Specialized Models are Better

Meta-tune a model that is initialized with T5-large and test it on unseen (non-similar) datasets 

```python3 default_train.py large```

Test UnifiedQA on all datatsets used for evaluation

```python3 baseline.py large```

Evaluate and compare the meta-tuned model and the UnifiedQA baseline with AUC-ROC for each label description. 

```python3 evaluate_and_plot.py large```

We should expect to see that meta-tuned model is better than the UnifiedQA model on the majority of label descriptions.

## Larger Models are Better

We can train another smaller-sized model using the command

```python3 default_train.py base```

and then we can compare the large vs. base model modifying  ```evaluate_and_plot.py```.

