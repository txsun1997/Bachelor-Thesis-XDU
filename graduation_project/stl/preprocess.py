import sys
import os
import torch
import time
import logging
from vocabulary import Vocabulary

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


def read_instances_from_file(files, max_len=400, keep_case=False):
    ''' Collect instances and construct vocab '''

    vocab = Vocabulary()
    lb_vocab = Vocabulary(need_default=False)
    sets = []

    for file in files:
        sents, labels = [], []
        trimmed_sent = 0
        with open(file) as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split('\t')
                if len(l) < 2:
                    continue
                label = l[0]
                sent = l[1]
                if not keep_case:
                    sent = sent.lower()
                word_lst = sent.split()
                if len(word_lst) > max_len:
                    word_lst = word_lst[:max_len]
                    trimmed_sent += 1
                if word_lst:
                    sents.append(word_lst)
                    labels.append(label)
                    vocab.add_word_lst(word_lst)
                    lb_vocab.add_word(label)

        assert len(sents) == len(labels)

        sets.append({
            'sents': sents,
            'labels': labels
        })

        logger.info('Get {} instances from file {}'.format(len(sents), file))
        if trimmed_sent:
            logger.info('{} sentences are trimmed. Max sentence length: {}.'
                        .format(trimmed_sent, max_len))

    logger.info('Building vocabulary...')
    vocab.add_word_lst(['<cls>'] * 6)
    vocab.build_vocab()
    lb_vocab.build_vocab()
    logger.info('Finished. Size of vocab: {}. # Class: {}.'.format(len(vocab), len(lb_vocab)))
    logger.info('<pad>: {}'.format(vocab.to_index('<pad>')))
    logger.info('<unk>: {}'.format(vocab.to_index('<unk>')))
    logger.info('<cls>: {}'.format(vocab.to_index('<cls>')))

    return sets, vocab, lb_vocab


def read_instances_from_test_file(test_file, max_len=400, keep_case=False):
    '''
    Collect instances from test file.
    Difference from above: do not add word to vocab.
    '''

    sents, labels = [], []
    trimmed_sent = 0
    with open(test_file) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split('\t')
            label = l[0]
            sent = l[1]
            if not keep_case:
                sent = sent.lower()
            word_lst = sent.split()
            if len(word_lst) > max_len:
                word_lst = word_lst[:max_len]
                trimmed_sent += 1
            if word_lst:
                sents.append(word_lst)
                labels.append(label)

    assert len(sents) == len(labels)

    test_set = {
        'sents': sents,
        'labels': labels
    }

    if trimmed_sent:
        logger.info('{} sentences are trimmed. Max sentence length: {}'
                    .format(trimmed_sent, max_len))

    logger.info('Get {} instances from file {}'.format(len(sents), test_file))

    return test_set


def convert_to_idx(sets, vocab, lb_vocab):
    ''' convert token into index using vocab '''

    idx_sets = []

    for set in sets:
        sents = set['sents']
        labels = set['labels']

        idx_sents = []
        idx_labels = []

        for i in range(len(sents)):
            idx_sent = [vocab.to_index('<cls>')]
            for word in sents[i]:
                idx_sent.append(vocab.to_index(word))
            idx_sents.append(idx_sent)
            idx_labels.append(lb_vocab.to_index(labels[i]))

        idx_sets.append({
            'sents': idx_sents,
            'labels': idx_labels
        })

    return idx_sets


def main():
    ''' main function '''

    start_time = time.time()

    data_path = '/remote-home/txsun/data/multi-task-cls/Product_all'
    dataset = {}
    for file in os.listdir(data_path):
        prefix = file.split('.')[0]
        if prefix in dataset:
            dataset[prefix].append(file)
        else:
            dataset[prefix] = [file]

    for k, v in dataset.items():
        # k: 'health'
        # v: ['health.dev', 'health.train', 'health.test']
        logger.info('Reading {}...'.format(k))
        assert len(v) == 3
        save_path = os.path.join('data', k)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
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
        text_sets, vocab, lb_vocab = read_instances_from_file([train_file, dev_file])
        text_test_set = read_instances_from_test_file(test_file)
        text_sets.append(text_test_set)
        idx_sets = convert_to_idx(text_sets, vocab, lb_vocab)
        train_set, dev_set, test_set = idx_sets

        data = {
            'vocab': vocab,
            'class_dict': lb_vocab,
            'train': train_set,
            'dev': dev_set,
            'test': test_set
        }

        logger.info('Testing pre-processing...')
        logger.info('The first two examples in train set:')
        logger.info(' '.join([vocab.to_word(idx) for idx in train_set['sents'][0]]))
        logger.info('label: {}'.format(lb_vocab.to_word(train_set['labels'][0])))

        logger.info(' '.join([vocab.to_word(idx) for idx in train_set['sents'][1]]))
        logger.info('label: {}'.format(lb_vocab.to_word(train_set['labels'][1])))

        logger.info('Dumping the processed data to pickle file {}'
                    .format(os.path.join(save_path, 'data.pkl')))
        torch.save(data, os.path.join(save_path, 'data.pkl'))

        logger.info('Finished. Dumping vocabulary to file {}'
                    .format(os.path.join(save_path, 'vocab.txt')))
        with open(os.path.join(save_path, 'vocab.txt'), mode='w', encoding='utf-8') as f:
            for i in range(len(vocab)):
                f.write(vocab.to_word(i) + '\n')

        logger.info('Finished. Dumping labels to file {}'
                    .format(os.path.join(save_path, 'labels.txt')))
        with open(os.path.join(save_path, 'labels.txt'), mode='w', encoding='utf-8') as f:
            for i in range(len(lb_vocab)):
                f.write(lb_vocab.to_word(i) + '\n')
        logger.info('Finished.')

    logger.info('Finished. Elapse: {}s.'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
