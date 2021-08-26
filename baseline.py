import pickle as pkl
import torch
from model import T5ZeroShotClfQA
import os
import sys
from path_utils import get_name_from_path_name, testing_dicts_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

size = sys.argv[1]
if __name__ == '__main__':
    model_name = "allenai/unifiedqa-t5-{size}".format(size=size)
    m = T5ZeroShotClfQA(model_name).to(device)
    eval_dataset_paths = [os.path.join(testing_dicts_dir, p) for p in os.listdir(testing_dicts_dir)]

    for eval_dataset_path in eval_dataset_paths:
        name = get_name_from_path_name(eval_dataset_path)
        label_q2input_answer = pkl.load(open(eval_dataset_path, 'rb'))
        label_q2preds = {}

        for (label, q), input_answer in label_q2input_answer.items():
            all_preds = m.get_logits_from_qc(input_answer)
            label_q2preds[(label, q)] = all_preds
        pkl.dump(label_q2preds, open('results/' + name + model_name, 'wb'))
