import argparse
import os
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from utils import load_word_emb
from dataset import ClsDataset, custom_collate
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

summary_writer = None
vocab = None
steps = 0
model_config = {}
config_str = ''
logger = logging.getLogger('train')
logger.setLevel(level=logging.DEBUG)


def train_epoch(model, train_iter, optimizer, args):
    global steps
    global summary_writer
    global config_str
    global model_config

    total_loss = 0
    corrects, samples = 0, 0

    print_every = 1

    model.train()

    for batch in train_iter:

        steps += 1
        # fetch one batch of data
        x, y, mask = batch
        x, y, mask = x.cuda(), y.cuda(), mask.cuda()
        # forward
        loss, pred = model(x, y, mask)
        # backward
        loss = loss / args.accumulation_steps
        loss.backward()
        total_loss += loss.item()
        # accuracy
        samples += x.size(0)
        corrects += (pred.data == y.data).sum()

        if steps % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            if steps % print_every == 0:
                summary_writer.add_scalar('Train_Loss/' + config_str, total_loss / print_every, steps)
                acc = float(corrects) / float(samples)
                summary_writer.add_scalar('Train_Acc/' + config_str, acc, steps)
                logger.info('  - Step {}. loss {}, acc {:.3f}%'.format(steps, total_loss, acc * 100))
                total_loss = 0
                corrects, samples = 0, 0

    return


def eval_epoch(model, data_iter, args):
    logger.info('Evaluating...')

    total_loss = 0
    esteps = 0
    corrects, samples = 0, 0

    model.eval()

    with torch.no_grad():
        for batch in data_iter:
            x, y, mask = batch
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()

            loss, pred = model(x, y, mask)

            total_loss += loss.item()
            esteps += 1

            samples += x.size(0)
            corrects += (pred.data == y.data).sum()

    avg_loss = total_loss / esteps
    acc = float(corrects) * 100.0 / float(samples)

    logger.info('Finished.')

    return avg_loss, acc


def train(model, train_iter, dev_iter, test_iter, optimizer, args):
    ''' start training '''

    total_time = time.time()
    logger.info('Training......')
    # model = nn.DataParallel(model, device).cuda()
    model = model.cuda()

    global config_str
    log_path = os.path.join(args.log_dir, args.dataset, config_str)

    while os.path.exists(log_path):
        log_path += '-x'
        config_str += '-x'
    global summary_writer
    summary_writer = SummaryWriter(log_path)

    for i_epoch in range(args.n_epoch):
        logger.info('Epoch {}'.format(i_epoch + 1))

        start_time = time.time()
        train_epoch(model, train_iter, optimizer, args)
        logger.info('Eposh {} finished. Elapse: {:.3f}s'
                    .format(i_epoch + 1, time.time() - start_time))

        dev_loss, dev_acc = eval_epoch(model, dev_iter, args)
        summary_writer.add_scalar('Validation_Loss/' + config_str, dev_loss, i_epoch + 1)
        summary_writer.add_scalar('Validation_Acc/' + config_str, dev_acc, i_epoch + 1)
        logger.info('Validation. loss {}, acc {:.3f}%'.format(dev_loss, dev_acc))

        test_loss, test_acc = eval_epoch(model, test_iter, args)
        summary_writer.add_scalar('Test_Loss/' + config_str, test_loss, i_epoch + 1)
        summary_writer.add_scalar('Test_Acc/' + config_str, test_acc, i_epoch + 1)
        logger.info('Test. loss {}, acc {:.3f}%'.format(test_loss, test_acc))

    summary_writer.close()
    del summary_writer

    logger.info('Training finished. Cost {:.3f} hours.'.format((time.time() - total_time) / 3600))
    logger.info('Dumping model...')
    save_path = os.path.join(args.save_path, args.dataset)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, config_str)
    torch.save(model, save_path + '.pkl')
    logger.info('Model saved as {}.'.format(save_path + '.pkl'))

    return


def main():
    ''' main function '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n_epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=50)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-accumulation_steps', type=int, default=1)
    parser.add_argument('-freeze', type=int, default=0)
    parser.add_argument('-same_lr', type=int, default=0)
    parser.add_argument('-dataset', type=str, default='sports')
    parser.add_argument('-model_config', type=str,
                        default='tf-6-4-512.config')
    parser.add_argument('-add_com', type=str,
                        default='stl')
    parser.add_argument('-log_dir', type=str,
                        default='/remote-home/txsun/fnlp/watchboard/product/stl')
    parser.add_argument('-save_path', type=str,
                        default='saved_models/')
    parser.add_argument('-embed_path', type=str,
                        default='/remote-home/txsun/data/word-embedding/glove/glove.840B.300d.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    bsz = args.batch_size // args.accumulation_steps

    global logger
    global model_config
    global config_str

    model_config = {}
    print('Reading configure file {}...'.format(args.model_config))
    with open(args.model_config, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split(':')[0].strip()
            value = line.split(':')[1].strip()
            model_config[key] = value
            print('{}: {}'.format(key, value))

    config_str = ''
    for key, value in model_config.items():
        config_str += key + '-' + value + '-'
    config_str += args.add_com

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # File Handler
    log_path = os.path.join('logs', args.dataset)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = os.path.join(log_path, config_str)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                       datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info('========== Loading Datasets ==========')
    dataset_file = os.path.join('data', args.dataset, 'data.pkl')
    logger.info('Loading dataset {}...'.format(dataset_file))
    data = torch.load(dataset_file)
    global vocab
    vocab = data['vocab']
    args.vocab_size = len(vocab)
    lb_vocab = data['class_dict']
    args.n_class = len(lb_vocab)
    logger.info('# classes: {}'.format(args.n_class))

    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    train_set = ClsDataset(train_data)
    train_iter = DataLoader(train_set, batch_size=bsz, drop_last=True,
                            shuffle=True, num_workers=2, collate_fn=custom_collate)
    logger.info('Train set loaded.')

    dev_set = ClsDataset(dev_data)
    dev_iter = DataLoader(dev_set, batch_size=args.batch_size,
                          num_workers=2, collate_fn=custom_collate)
    logger.info('Development set loaded.')

    test_set = ClsDataset(test_data)
    test_iter = DataLoader(test_set, batch_size=args.batch_size,
                           num_workers=2, collate_fn=custom_collate)
    logger.info('Test set loaded.')
    logger.info('Datasets finished.')

    logger.info('====== Loading Word Embedding =======')
    we_path = os.path.join('data', args.dataset, 'word_embedding.npy')
    word_embedding = load_word_emb(args.embed_path, 300, vocab, save_path=we_path)

    logger.info('========== Preparing Model ==========')
    model = Transformer(args, model_config, word_embedding)

    logger.info('Model parameters:')
    params = list(model.named_parameters())
    sum_param = 0
    for name, param in params:
        if param.requires_grad == True:
            logger.info('{}: {}'.format(name, param.shape))
            sum_param += param.numel()
    logger.info('# Parameters: {}.'.format(sum_param))

    logger.info('========== Training Model ==========')
    lr = float(model_config['lr'])
    if args.same_lr or args.freeze:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr)
    else:
        word_embed_params = list(map(id, model.embed.word_embeddings.parameters()))
        base_params = filter(lambda p: id(p) not in word_embed_params, model.parameters())
        opt = optim.Adam([
            {'params': base_params},
            {'params': model.embed.word_embeddings.parameters(), 'lr': lr * 0.1}
        ], lr=lr)

    train(model, train_iter, dev_iter, test_iter, opt, args)

    return


if __name__ == '__main__':
    main()
