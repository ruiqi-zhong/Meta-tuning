import pickle as pkl
import random
from tqdm import trange
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from model import T5ZeroShotClfQA
from path_utils import get_name_from_path_name, get_id_from_path_name, testing_dicts_dir, training_dicts_dir
import os
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, dataset_paths, save_every, steps, save_path, bsize):
    data_dicts = []
    for d_path in dataset_paths:
        data_dicts.extend(pkl.load(open(d_path, 'rb')))
    print('%d datapoints' % len(data_dicts))
    random.shuffle(data_dicts)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, steps)
    loss_func = torch.nn.NLLLoss()

    for i in trange(steps):
        ds = [data_dicts[j % len(data_dicts)] for j in range(i * bsize, (i + 1) * bsize)]
        logits = model.get_logits_from_qc_(ds)
        gold = torch.tensor([d['a'] for d in ds]).to(device)
        loss = loss_func(logits, gold)
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % save_every == 0 and i != 0:
            model.save(save_path + str(i))


size = sys.argv[1]
steps = 5001
if __name__ == '__main__':
    groups = {get_id_from_path_name(p) for p in os.listdir(testing_dicts_dir)}

    for group_id in groups:
        train_dataset_paths = [os.path.join(training_dicts_dir, p) for p in os.listdir(training_dicts_dir) if get_id_from_path_name(p) != group_id]
        eval_dataset_paths = [os.path.join(testing_dicts_dir, p) for p in os.listdir(testing_dicts_dir) if get_id_from_path_name(p) == group_id]

        config_name = 'group%d' % group_id

        lock_path = 'locks/t5init%s%s' % (size, config_name)
        if os.path.exists(lock_path):
            continue
        pkl.dump('lock', open(lock_path, 'wb'))

        train_names = [get_name_from_path_name(p) for p in train_dataset_paths]
        eval_names = [get_name_from_path_name(p) for p in eval_dataset_paths]
        print('Train on ', train_names)
        print('Evaluate on', eval_names)

        model_name = "t5-%s" % size
        m = T5ZeroShotClfQA(model_name).to(device)
        train(m, train_dataset_paths, steps - 1, steps, 'checkpoints/%s%s' % (size, config_name), 32)

        ckpt_path = 'checkpoints/%s%s%d' % (size, config_name, steps - 1)
        m.load(ckpt_path)
        for eval_dataset_path in eval_dataset_paths:
            name = get_name_from_path_name(eval_dataset_path)
            label_q2input_answer = pkl.load(open(eval_dataset_path, 'rb'))
            label_q2preds = {}

            for (label, q), input_answer in label_q2input_answer.items():
                # input_answer is a list of datapoints, each is a dictionary with "q", "c", "a" keys.
                # all_preds is a 2D logit, with dimension num_datapoints X 2.
                # The semantics for the 2nd dimension is [logit for "no", logit for "yes"].
                all_preds = m.get_logits_from_qc(input_answer)
                label_q2preds[(label, q)] = all_preds
            pkl.dump(label_q2preds, open('results/' + name + model_name, 'wb'))
        os.unlink(ckpt_path)
