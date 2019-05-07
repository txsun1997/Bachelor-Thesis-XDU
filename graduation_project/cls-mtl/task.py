from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler, SequentialSampler


class Task(object):

    def __init__(self, task_id, task_name, train_set, dev_set, test_set):

        self.task_id = task_id
        self.task_name = task_name

        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set

        self.train_data_loader = None
        self.dev_data_loader = None
        self.test_data_loader = None

    def init_data_loader(self, batch_size):

        self.train_data_loader = Batch(self.train_set, batch_size, sampler=RandomSampler())
        self.train_data_loader.init_iter()
        self.dev_data_loader = Batch(self.dev_set, batch_size, sampler=SequentialSampler())
        self.test_data_loader = Batch(self.test_set, batch_size, sampler=SequentialSampler())