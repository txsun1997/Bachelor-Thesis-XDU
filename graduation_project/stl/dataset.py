import torch
import torch.utils.data
import numpy as np


def custom_collate(batch):
    ''' padding '''

    DEFAULT_PADDING_LABEL = 0

    sents, labels = zip(*batch)
    max_len = max(len(sent) for sent in sents)

    batch_sent = [sent + [DEFAULT_PADDING_LABEL]\
                  * (max_len - len(sent)) for sent in sents]

    batch_sent = torch.LongTensor(np.array(batch_sent))
    batch_label = torch.LongTensor(np.array(labels))
    batch_mask = (batch_sent != DEFAULT_PADDING_LABEL)

    return batch_sent, batch_label, batch_mask


class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.sent = dataset['sents']
        self.label = dataset['labels']

    def __getitem__(self, item):
        return self.sent[item], self.label[item]

    def __len__(self):
        return len(self.sent)