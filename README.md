# Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections

Ruiqi Zhong, Kristy Lee<sup>\*</sup>, Zheng Zhang<sup>\*</sup>, Dan Klein

EMNLP 2021 Findings, https://arxiv.org/abs/2104.04670


## Released Model Checkpoint and How to Use It

We uploaded meta-tuned roberta-large to the huggingface model hub as ```ruiqi-zhong/roberta-large-meta-tuning-test```.
Example code can be seen in ```demo.py``` . 
We also uploaded meta-tuned roberta-base as well. 
Note: please do not compare its performance with other zero-shot learning models, since it has seen all the datasets in our paper during training time. 

## Data

Please download the dataset from here: https://drive.google.com/file/d/1hrLlpk6Pla95Bnv_e1MAhCx7uJSDgA-w/view?usp=sharing  

If you are using this dataset, please cite all the papers in the custom_citations.txt, anthology_citations.txt, urls.txt file in the citations folder. Thanks!

```training_dicts/``` contains all the datasets for training, and each of the .pkl file is a list of datapoints. 
Each datapoint is represented as a dictionary.

```{"q": [label description], "c": [text input], "a": [0 or 1]}```, 

where "q" stands for question, which contains label information, "c" stands for context, which contains the input text, "a" stands for answer, which is either 1 (Yes) or 0 (No). 

```testing_dicts/``` contains all the datasets for evaluation, and each of the .pkl file is a map from ```(label_index, label descriptions "q")``` to a list of datapoints.

```label_index``` is a unique integer associated with a particular label in the dataset. for example, in topic classification, ```sports=0, global_news=1, education=2 ...``` etc.

Datasets that have the same group number in front of their filenames are considered similar. 
Notice that,  for each dataset, there might be overlapping datapoints between the training and testing split, but it is okay since we never train and test on the same dataset.

Additionally, to speedup evaluation, we performed subsampling for many of the test datasets, so the numbers will not be directly comparable to those in the other paper.

## Specialized Models are Better

Meta-tune a model that is initialized with T5-large and test it on unseen (non-similar) datasets 

```python3 default_train.py large```

Test UnifiedQA on all datatsets used for evaluation

```python3 baseline.py large```

Evaluate and compare the meta-tuned model and the UnifiedQA baseline with AUC-ROC for each label description. 

```python3 evaluate_and_plot.py large```

We should expect to see that meta-tuned model is better than the UnifiedQA model on the majority of label descriptions (as shown in the ```metatunedvsunifiedqa-large.png```). 
The figure will not be exactly the same as the one you get or the one shown in the figure (Figure 6.a) due to random seeds, but the qualitative conclusion should be robust.

## Larger Models are Better

We can train another smaller-sized model using the command

```python3 default_train.py base```

and then we can compare the large vs. base model modifying  ```evaluate_and_plot.py```.

