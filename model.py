import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import numpy as np
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_count = torch.cuda.device_count()


def normalize(t):
    return re.sub("'(.+)'", r'\1', t.lower())


def qc2input(d):
    return normalize(d['q'] + '\\n' + d['c'])


class T5ZeroShotClfQA(torch.nn.Module):

    def __init__(self, qa_model_name, max_seq_length = 128):
        super(T5ZeroShotClfQA, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(qa_model_name)
        if device == 'cuda':
            num_attention_modules = self.model.config.num_layers

            device_map = {i: list(range(i * num_attention_modules // device_count, (i + 1) * num_attention_modules // device_count))
                          for i in range(device_count)}
            self.model.parallelize(device_map)
        self.vocab = self.tokenizer.get_vocab()
        self.yes_id, self.no_id = self.vocab['▁yes'], self.vocab['▁no']
        self.max_seq_length = max_seq_length
        self.lsm = torch.nn.LogSoftmax(dim=-1)

    def create_batch(self, q_dicts):
        input_strings = [qc2input(d) for d in q_dicts]
        input_strings = [normalize(i) for i in input_strings]
        input_dict = self.tokenizer(input_strings, padding=True, return_tensors="pt",
                                    truncation=True, max_length=self.max_seq_length).to(device)
        return input_dict

    def forward(self, input_dict):
        starts = torch.tensor([[self.model.config.decoder_start_token_id]] * len(input_dict['input_ids'])).to(device)
        output = self.model(**input_dict, decoder_input_ids=starts)
        logits = self.lsm(output.logits[:, 0, [self.no_id, self.yes_id]])
        return logits

    def get_logits_from_qc_(self, input_strings):
        input_dict = self.create_batch(input_strings)
        return self.forward(input_dict)

    def get_logits_from_qc(self, q_dicts, bsize=32, progress_bar=True):
        self.model.eval()
        result_logits = []
        iter_count = (len(q_dicts) - 1) // bsize + 1
        ranger = range(iter_count) if not progress_bar else tqdm.trange(iter_count)
        for i in ranger:
            l = self.get_logits_from_qc_(q_dicts[i*bsize:(i+1) * bsize]).detach().cpu().numpy().tolist()
            result_logits.extend(l)
        return np.array(result_logits)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

