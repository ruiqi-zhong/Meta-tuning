import sys
import os
from sklearn.metrics import roc_auc_score, roc_curve
from path_utils import testing_dicts_dir, get_name_from_path_name
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

eval_dataset_paths = [os.path.join(testing_dicts_dir, p) for p in os.listdir(testing_dicts_dir)]
size = sys.argv[1]


def calculate_task2label2scores(suffix):
    task2results = {}
    for p in eval_dataset_paths:
        name = get_name_from_path_name(p)
        label_q2inputs = pkl.load(open(p, 'rb'))
        label_q2preds = pkl.load(open('results/%s%s' % (name, suffix), 'rb'))

        result, q_map = {}, {}
        for (label, q) in sorted(label_q2inputs):
            if q not in q_map:
                q_map[q] = len(q_map)

        for (label, q) in label_q2inputs:
            yes_no_gold = np.array([d['a'] for d in label_q2inputs[(label, q)]])
            preds = label_q2preds[(label, q)][:, 1]
            result[(label, q_map[q])] = roc_auc_score(yes_no_gold, preds)

        task2results[name] = result
    return task2results


baseline_name = 'allenai-unifiedqa-t5-%s' % size
treatment_name = 't5-%s' % size

baseline = calculate_task2label2scores(baseline_name)
metatuned = calculate_task2label2scores(treatment_name)

xs, ys = [], []
for task, results in baseline.items():
    for k in results:
        xs.append(baseline[task][k])
        ys.append(metatuned[task][k])

plt.scatter(xs, ys, alpha=0.3, color='b')
plt.plot(np.arange(0.3, 1, 0.01), np.arange(0.3, 1, 0.01), color='r')
plt.xlabel(baseline_name)
plt.ylabel(treatment_name)
plt.title('AUC ROC scores for each label descriptions')
plt.savefig('metatunedvsunifiedqa-%s' % size)

