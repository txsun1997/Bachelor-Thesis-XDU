import sys
import os
import time
import pickle
import logging
from fastNLP import Vocabulary
from task import Task
from utils import load_word_emb
from fastNLP import DataSet
from fastNLP import Instance


def read_instances_from_file(file, max_len=400, keep_case=False):
    ''' Collect instances and construct vocab '''

    dataset = DataSet()
    trimmed_sent = 0

    with open(file) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split('\t')
            if len(l) < 2:
                continue
            label = int(l[0])
            sent = l[1]
            if not keep_case:
                sent = sent.lower()
            word_lst = sent.split()
            if len(word_lst) > max_len:
                word_lst = word_lst[:max_len]
                trimmed_sent += 1
            if word_lst:
                dataset.append(Instance(words=word_lst, label=label))

    logger.info('Get {} instances from file {}'.format(len(dataset), file))
    if trimmed_sent:
        logger.info('{} sentences are trimmed. Max sentence length: {}.'
                    .format(trimmed_sent, max_len))

    return dataset


if __name__ == '__main__':

    logger = logging.getLogger('preprocess')
    logger.setLevel(level=logging.DEBUG)

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # File Handler
    file_handler = logging.FileHandler('logs/preprocess.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                       datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    start_time = time.time()

    data_path = '/remote-home/txsun/data/multi-task-cls/Product_all'
    dataset = {}
    task_lst = []
    vocab = Vocabulary()

    for file in os.listdir(data_path):
        prefix = file.split('.')[0]
        if prefix in {'topic', 'sembedding'}:
            continue
        if prefix in dataset:
            dataset[prefix].append(file)
        else:
            dataset[prefix] = [file]

    logger.info('# of Dataset: {}'.format(len(dataset)))

    for task_id, (k, v) in enumerate(dataset.items()):
        # k: 'health'
        # v: ['health.dev', 'health.train', 'health.test']
        logger.info('Reading {}...'.format(k))
        assert len(v) == 3
        train_file = None
        dev_file = None
        test_file = None
        for f in v:
            # example of f: 'health.dev'
            data_type = f.split('.')[1]
            if data_type == 'train':
                train_file = os.path.join(data_path, f)
            elif data_type == 'dev':
                dev_file = os.path.join(data_path, f)
            elif data_type == 'test':
                test_file = os.path.join(data_path, f)
            else:
                raise ValueError('unknown dataset type')
        train_set = read_instances_from_file(train_file)
        train_set.add_field('task_id', [task_id] * len(train_set))
        train_set.apply(lambda x: vocab.add_word_lst(x['words']))

        dev_set = read_instances_from_file(dev_file)
        dev_set.add_field('task_id', [task_id] * len(dev_set))
        dev_set.apply(lambda x: vocab.add_word_lst(x['words']))

        test_set = read_instances_from_file(test_file)
        test_set.add_field('task_id', [task_id] * len(test_set))
        # test_set.apply(lambda x: vocab.add_word_lst(x['words']))

        task = Task(task_id, k, train_set, dev_set, test_set)
        task_lst.append(task)

    logger.info('Building vocabulary...')
    vocab.build_vocab()
    logger.info('Finished. Size of vocab: {}.'.format(len(vocab)))
    for task in task_lst:
        task.train_set.apply(lambda x: [vocab.to_index(w) for w in x['words']],
                             new_field_name='words_idx')

        task.dev_set.apply(lambda x: [vocab.to_index(w) for w in x['words']],
                           new_field_name='words_idx')

        task.test_set.apply(lambda x: [vocab.to_index(w) for w in x['words']],
                            new_field_name='words_idx')

        task.train_set.set_input('task_id', 'words_idx', flag=True)
        task.train_set.set_target('label', flag=True)

        task.dev_set.set_input('task_id', 'words_idx', flag=True)
        task.dev_set.set_target('label', flag=True)

        task.test_set.set_input('task_id', 'words_idx', flag=True)
        task.test_set.set_target('label', flag=True)

    logger.info('Finished. Dumping vocabulary to data/vocab.txt')
    with open('data/vocab.txt', mode='w', encoding='utf-8') as f:
        for i in range(len(vocab)):
            f.write(vocab.to_word(i) + '\n')

    logger.info('Testing data...')
    for task in task_lst:
        logger.info(str(task.task_id) + ' ' + task.task_name)
        logger.info(task.train_set[0])
        logger.info(task.dev_set[0])
        logger.info(task.test_set[0])

    logger.info('Dumping data...')
    data = {'task_lst': task_lst}
    save_file = open('data/data.pkl', 'wb')
    pickle.dump(data, save_file)
    save_file.close()
    logger.info('Finished. Looking up for word embeddings...')
    embed_path = '/remote-home/txsun/data/word-embedding/glove/glove.840B.300d.txt'
    _ = load_word_emb(embed_path, 300, vocab)
    logger.info('Finished. Elapse: {}s.'.format(time.time() - start_time))
    logger.removeHandler(stream_handler)
    logger.removeHandler(file_handler)
