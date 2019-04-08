import os
import logging
import numpy as np
import torch.nn as nn
import torch.nn.init as init


def load_word_emb(path, embed_dim, vocab, save_path=None):
    ''' load word embedding from file '''

    logger = logging.getLogger('preprocess.load_word_emb')
    if save_path is None:
        save_path = 'data/word_embedding.npy'
    if os.path.exists(save_path):
        logger.info('Loading existed word embeddings from {}.'.format(save_path))
        word_embedding = np.load(save_path)
        logger.info('Word embedding finished.')
        return word_embedding

    word_embedding = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))
    word_embedding[0] = np.zeros((1, embed_dim))

    occur_word = 0
    with open(path, encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split(' ')
            word = values[0]
            idx = vocab.to_index(word)
            if idx == 1:
                continue
            else:
                occur_word += 1
                for i in range(embed_dim):
                    word_embedding[idx][i] = float(values[i + 1])

    oov = len(vocab) - occur_word
    logger.info('Pre-trained word embeddings loaded. OOV: {}({:.2f}%)'.format(oov, oov * 100 / len(vocab)))
    logger.info('Dumping pre-trained word embeddings in {}.'.format(save_path))
    np.save(save_path, word_embedding)
    logger.info('Word embedding finished.')
    return word_embedding


def initial_parameter(net, initial_method=None):
    """A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model
    :param initial_method: str, one of the following initializations

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif hasattr(m, 'weight') and m.weight.requires_grad:
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    net.apply(weights_init)
