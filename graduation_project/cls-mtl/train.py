import os
import sys
import time
import pickle
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from transformer import Transformer
# from tf_stack_mean_pooling import Transformer
# from tf_stack_cls import Transformer
# from tf_adjoint_implicit import Transformer
from tf_adjoint_explicit import Transformer
from tensorboardX import SummaryWriter


def find_task(task_id, task_lst):
    if task_lst[task_id].task_id == task_id:
        return task_lst[task_id]
    for task in task_lst:
        if task_id == task.task_id:
            return task
    raise RuntimeError('Cannot find task with task_id={}.'.format(task_id))


class Trainer(object):

    def __init__(self, model, description, task_lst, optimizer, log_path, save_path,
                 accumulation_steps, print_every):
        '''
        :param model: 模型
        :param description: 模型描述
        :param task_lst: 任务列表
        :param optimizer: 优化器
        :param log_path: TensorboardX存储文件夹
        :param save_path: 模型存储位置
        :param accumulation_steps: 累积梯度
        :param print_every: 评估间隔
        '''
        self.model = model
        self.task_lst = task_lst
        self.save_path = save_path
        self.description = description
        self.optim = optimizer
        self.accumulation_steps = accumulation_steps
        self.print_every = print_every

        self.steps = 0
        self.best_acc = 0
        self.best_epoch = 0

        self.logger = logging.getLogger('train')
        self.logger.setLevel(level=logging.DEBUG)

        self.summary_writer = SummaryWriter(os.path.join(log_path, description))

        # Stream Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        stream_handler.setFormatter(stream_formatter)
        self.logger.addHandler(stream_handler)

        # File Handler
        logger_path = os.path.join('logs', description)
        file_handler = logging.FileHandler(logger_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                           datefmt='%Y/%m/%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def train(self, n_epoch):
        total_time = time.time()
        self.logger.info('Start training...')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        for i_epoch in range(n_epoch):
            start_time = time.time()
            self.cur_epoch = i_epoch
            self.logger.info('Epoch {}'.format(i_epoch))
            self._train_epoch()
            self.logger.info('Epoch {} finished. Elapse: {:.3f}s.'
                             .format(i_epoch, time.time() - start_time))

            dev_loss, dev_acc = self._eval_epoch()
            self.summary_writer.add_scalar('dev_loss/' + self.description, dev_loss, i_epoch)
            self.summary_writer.add_scalars('dev_acc/' + self.description, {
                'AVG': dev_acc['avg'],
                'apparel': dev_acc['apparel'],
                'baby': dev_acc['baby'],
                'books': dev_acc['books'],
                'camera': dev_acc['camera'],
                'dvd': dev_acc['dvd'],
                'elec': dev_acc['elec'],
                'health': dev_acc['health'],
                'imdb': dev_acc['imdb'],
                'kitchen': dev_acc['kitchen'],
                'mag': dev_acc['mag'],
                'mr': dev_acc['mr'],
                'music': dev_acc['music'],
                'soft': dev_acc['soft'],
                'sports': dev_acc['sports'],
                'toys': dev_acc['toys'],
                'video': dev_acc['video']
            }, i_epoch)
            self.logger.info('Validation loss {}, avg acc {:.3f}%'
                             .format(dev_loss, dev_acc['avg']))
            if dev_acc['avg'] > self.best_acc:
                self.best_acc = dev_acc['avg']
                self.best_epoch = i_epoch
                self.logger.info('Updating best model...')
                self._save_model()
                self.logger.info('Model saved.')

            self.logger.info('Current best acc [{:.3f}%] occured at epoch [{}].'
                             .format(self.best_acc, self.best_epoch))
        self.logger.info('Training finished. Elapse {:.3f} hours.'
                         .format((time.time() - total_time) / 3600))

    def _train_epoch(self):

        total_loss = 0
        corrects, samples = 0, 0

        n_tasks = len(self.task_lst)
        task_seq = list(np.random.permutation(n_tasks))
        empty_task = set()
        self.model.train()

        while len(empty_task) < n_tasks:
            for task_id in task_seq:
                if task_id in empty_task:
                    continue
                task = find_task(task_id, self.task_lst)
                batch = task.train_data_loader.fetch_one()
                if batch is None:
                    empty_task.add(task_id)
                    task.train_data_loader.init_iter()
                    continue
                x, y = batch
                batch_task_id = x['task_id'].cuda()
                batch_x = x['words_idx'].cuda()
                batch_y = y['label'].cuda()

                loss, pred = self.model(batch_task_id, batch_x, batch_y)
                self.steps += 1

                total_loss += loss.item()
                loss = loss / self.accumulation_steps
                loss.backward()

                samples = batch_x.size(0)
                corrects += (pred.data == batch_y.data).sum()

                if self.steps % self.accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    self.optim.zero_grad()

                if self.steps % self.print_every == 0:
                    self.summary_writer.add_scalar('train_loss/' + self.description, total_loss / self.print_every,
                                                   self.steps)
                    acc = float(corrects) / float(samples)
                    self.summary_writer.add_scalar('train_acc/' + self.description, acc * 100, self.steps)
                    self.logger.info(' - Step {}: loss {} acc {:.3f}%'
                                     .format(self.steps, total_loss / self.print_every, acc * 100))
                    total_loss = 0
                    corrects, samples = 0, 0

    def _eval_epoch(self):
        self.logger.info('Evaluating...')
        dev_loss = 0
        e_steps = 0
        avg_acc = 0
        dev_acc = {}
        self.model.eval()

        with torch.no_grad():
            for i in range(len(self.task_lst)):
                corrects, samples = 0, 0
                task = find_task(i, self.task_lst)
                for batch in task.dev_data_loader:
                    x, y = batch
                    batch_task_id = x['task_id'].cuda()
                    batch_x = x['words_idx'].cuda()
                    batch_y = y['label'].cuda()

                    loss, pred = self.model(batch_task_id, batch_x, batch_y)
                    dev_loss += loss.item()
                    e_steps += 1

                    samples += batch_x.size(0)
                    corrects += (pred.data == batch_y.data).sum()

                acc = float(corrects) * 100.0 / float(samples)
                avg_acc += acc
                dev_acc[task.task_name] = acc

        avg_acc /= len(self.task_lst)
        dev_acc['avg'] = avg_acc
        dev_loss = dev_loss / e_steps
        return dev_loss, dev_acc

    def _save_model(self):
        save_path = os.path.join(self.save_path, self.description)
        self.model.cpu()
        torch.save(self.model, save_path)
        self.model.cuda()


def test_model(model, task_lst):
    if torch.cuda.is_available():
        model = model.cuda()
    loss = 0
    steps = 0
    avg_acc = 0
    test_acc = {}
    model.eval()

    with torch.no_grad():
        for i in range(len(task_lst)):
            corrects, samples = 0, 0
            task = find_task(i, task_lst)
            for batch in task.test_data_loader:
                x, y = batch
                batch_task_id = x['task_id'].cuda()
                batch_x = x['words_idx'].cuda()
                batch_y = y['label'].cuda()

                loss, pred = model(batch_task_id, batch_x, batch_y)
                loss += loss.item()
                steps += 1

                samples += batch_x.size(0)
                corrects += (pred.data == batch_y.data).sum()

            acc = float(corrects) * 100.0 / float(samples)
            avg_acc += acc
            test_acc[task.task_name] = acc

    avg_acc /= len(task_lst)
    test_acc['avg'] = avg_acc
    loss = loss / steps
    return loss, test_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n_epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=50)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-accumulation_steps', type=int, default=1)
    parser.add_argument('-print_every', type=int, default=1)
    parser.add_argument('-freeze', type=int, default=0)
    parser.add_argument('-n_class', type=int, default=2)
    parser.add_argument('-same_lr', type=int, default=0)
    parser.add_argument('-model_config', type=str,
                        default='tf-6-4-512.config')
    parser.add_argument('-model_descript', type=str,
                        default='stack_mean_pooling')
    parser.add_argument('-log_dir', type=str,
                        default='/remote-home/txsun/fnlp/watchboard/product/mtl')
    parser.add_argument('-save_path', type=str,
                        default='saved_models/')
    parser.add_argument('-embed_path', type=str,
                        default='/remote-home/txsun/data/word-embedding/glove/glove.840B.300d.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_config = {}
    print('Reading configure file {}...'.format(args.model_config))
    with open(args.model_config, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split(':')[0].strip()
            value = line.split(':')[1].strip()
            model_config[key] = value
            print('{}: {}'.format(key, value))
    print('========== Loading Datasets ==========')
    data_file = open('data/data.pkl', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    task_lst = data['task_lst']
    print('# of Tasks: {}.'.format(len(task_lst)))
    for task in task_lst:
        print('Task {}: {}'.format(task.task_id, task.task_name))
    bsz = args.batch_size // args.accumulation_steps
    for task in task_lst:
        task.init_data_loader(bsz)
    print('done.')

    print('====== Loading Word Embedding =======')
    word_embedding = np.load('data/word_embedding.npy')
    args.vocab_size = word_embedding.shape[0]
    print('vocab size: {}.'.format(args.vocab_size))
    print('done.')

    print('========== Preparing Model ==========')
    model = Transformer(args, model_config, word_embedding)

    print('Model parameters:')
    params = list(model.named_parameters())
    sum_param = 0
    for name, param in params:
        if param.requires_grad == True:
            print('{}: {}'.format(name, param.shape))
            sum_param += param.numel()
    print('# Parameters: {}.'.format(sum_param))

    print('========== Training Model ==========')
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

    trainer = Trainer(model, args.model_descript, task_lst, opt, args.log_dir, args.save_path,
                      args.accumulation_steps, args.print_every)

    trainer.train(args.n_epoch)

    print('========== Testing Model ==========')
    model = torch.load(os.path.join(args.save_path, args.model_descript))
    test_loss, test_acc = test_model(model, task_lst)
    print(args.model_descript)
    for acc in test_acc.items():
        print(acc)
