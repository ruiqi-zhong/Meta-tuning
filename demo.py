import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification
import re
import numpy as np
import tqdm

# change device to GPU if you have one
device = 'cpu'


def normalize(t):
    return re.sub("'(.+)'", r'\1', t.lower())


def qc2input(d):
    # We used the same input format as UnifiedQA
    return normalize(d['q'] + '\\n' + d['c'])


class BERTZeroShotClfQA(torch.nn.Module):

    # NOTICE THAT WE ONLY TRAINED THE MODEL FOR CONTEXT LENGTH 128
    def __init__(self, model_name, max_seq_length=128):
        super(BERTZeroShotClfQA, self).__init__()
        if max_seq_length > 128:
            raise Exception('We only trained our model for context length 128. '
                            'Feel free to remove this if you are training your own model.')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.max_seq_length = max_seq_length
        self.lsm = torch.nn.LogSoftmax(dim=-1)

    # first get the string formatted representation from the input data dictionary
    # then tokenize it create the model input dictionary
    def create_batch(self, q_dicts):
        input_strings = [qc2input(d) for d in q_dicts]
        input_dict = self.tokenizer(input_strings, padding=True, return_tensors="pt",
                                    truncation=True, max_length=self.max_seq_length).to(device)
        return input_dict

    def forward(self, input_dict):
        output = self.model(**input_dict)
        return self.lsm(output.logits)

    # returns a pytorch tensor given a datapoint dictionary
    def get_logits_from_qc_(self, datapoint_dicts):
        input_dict = self.create_batch(datapoint_dicts)
        return self.forward(input_dict)

    def get_logits_from_qc(self, q_dicts, bsize=32, progress_bar=True):
        self.model.eval()
        result_logits = []
        # perform inference by batches
        iter_count = (len(q_dicts) - 1) // bsize + 1
        ranger = range(iter_count) if not progress_bar else tqdm.trange(iter_count)
        for i in ranger:
            l = self.get_logits_from_qc_(q_dicts[i*bsize:(i+1) * bsize]).detach().cpu().numpy().tolist()
            result_logits.extend(l)
        return np.array(result_logits)


if __name__ == '__main__':
    # the input data format
    # each datapoint is represented as a dictionary
    # q the question describing the label
    # c the "context", which is the input to be classified
    # here are some simple examples
    data_dicts = [
        {'q': 'Does the user like this movie?', 'c': 'Great movie! I love it.'},
        {'q': 'Does the user like this movie?', 'c': 'Horrible movie. Total waste of my time.'},
        {'q': 'Does the user like this movie?', 'c': 'I would really recommend it to my friends!'},
        {'q': 'Does the user dislike this movie?', 'c': 'I don\'t like this movie.'}
    ]

    # loading the model
    model = BERTZeroShotClfQA('ruiqi-zhong/roberta-large-meta-tuning-test')

    # get_logits_from_qc returns numpy array
    logits = model.get_logits_from_qc(data_dicts)
    assert logits.shape == (len(data_dicts), 2)

    pred = np.argmax(logits, axis=-1)
    # predicted labels, 1 for yes, 0 for no.
    print('Predicted labels', pred)
