import argparse
import logging
import os
import sys
import pickle
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

import numpy as np
import torch

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler, SequentialSampler

logger = logging.getLogger('StlFineTuning')
logger.setLevel(level=logging.DEBUG)


def train(model, train_dataloader, dev_dataloader, optimizer, args):
    global_step = 0
    summary_writer = SummaryWriter(os.path.join(args.log_dir, args.model_descript))

    model.train()
    for i_epoch in trange(args.n_epoch, desc='Epoch'):
        tr_loss = 0
        tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            x, y = batch
            input_ids = x['input_ids'].cuda()
            input_mask = (input_ids != 0).cuda()
            segment_ids = x['segment_ids'].cuda()
            label_ids = y['label_id'].cuda()
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if args.n_gpu > 1:
                loss = loss.mean()
            loss = loss / args.accumulation_steps
            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                tr_steps += 1
                if global_step % args.print_every == 0:
                    summary_writer.add_scalar('Train/Loss/' + args.model_descript, tr_loss / tr_steps, global_step)
                    logger.info('Steps {}: Loss {:.4f}.'.format(global_step, tr_loss / tr_steps))
                    tr_loss = 0
                    tr_steps = 0
        # evaluation
        logger.info('Epoch {} finished. Start evaluating...'.format(i_epoch))
        model.eval()
        if args.task == 'MNLI':
            dev_matched_dataloader, dev_mismatched_dataloader = dev_dataloader
            # matched
            eval_matched_loss, eval_matched_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for x, y in tqdm(dev_matched_dataloader, desc="Evaluating"):
                input_ids = x['input_ids'].cuda()
                input_mask = (input_ids != 0).cuda()
                segment_ids = x['segment_ids'].cuda()
                label_ids = y['label_id'].cuda()

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.cpu().numpy()
                pred = np.argmax(logits, axis=1)
                tmp_eval_accuracy = np.sum(pred == label_ids)

                eval_matched_loss += tmp_eval_loss.mean().item()
                eval_matched_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_matched_loss = eval_matched_loss / nb_eval_steps
            eval_matched_accuracy = eval_matched_accuracy / nb_eval_examples

            # mismatched
            eval_mismatched_loss, eval_mismatched_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for x, y in tqdm(dev_mismatched_dataloader, desc="Evaluating"):
                input_ids = x['input_ids'].cuda()
                input_mask = (input_ids != 0).cuda()
                segment_ids = x['segment_ids'].cuda()
                label_ids = y['label_id'].cuda()

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.cpu().numpy()
                pred = np.argmax(logits, axis=1)
                tmp_eval_accuracy = np.sum(pred == label_ids)

                eval_mismatched_loss += tmp_eval_loss.mean().item()
                eval_mismatched_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_mismatched_loss = eval_mismatched_loss / nb_eval_steps
            eval_mismatched_accuracy = eval_mismatched_accuracy / nb_eval_examples

            # record
            summary_writer.add_scalars('Validation/Loss/' + args.model_descript, {
                'matched_loss': eval_matched_loss,
                'mismatched_loss': eval_mismatched_loss,
                'avg_loss': (eval_matched_loss + eval_mismatched_loss) / 2
            }, i_epoch)
            summary_writer.add_scalars('Validation/Acc/' + args.model_descript, {
                'matched_acc': eval_matched_accuracy,
                'mismatched_acc': eval_mismatched_accuracy,
                'avg_acc': (eval_matched_accuracy + eval_mismatched_accuracy) / 2
            }, i_epoch)

            logger.info('Matched loss: {}.'.format(eval_matched_loss))
            logger.info('Mismatched loss: {}.'.format(eval_mismatched_loss))
            logger.info('Matched acc: {}.'.format(eval_matched_accuracy))
            logger.info('Mismatched acc: {}.'.format(eval_mismatched_accuracy))
        else:
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for x, y in tqdm(dev_dataloader, desc="Evaluating"):
                input_ids = x['input_ids'].cuda()
                input_mask = (input_ids != 0).cuda()
                segment_ids = x['segment_ids'].cuda()
                label_ids = y['label_id'].cuda()

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.cpu().numpy()
                pred = np.argmax(logits, axis=1)
                tmp_eval_accuracy = np.sum(pred == label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            # record
            summary_writer.add_scalar('Validation/Loss/' + args.model_descript, eval_loss, i_epoch)
            summary_writer.add_scalar('Validation/Acc/' + args.model_descript, eval_accuracy, i_epoch)
            logger.info('loss: {}.'.format(eval_loss))
            logger.info('acc: {}.'.format(eval_accuracy))

        # save model
        model_to_save = model.module if hasattr(model, 'module') else model
        save_path = os.path.join(args.save_path, args.model_descript)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, str(i_epoch))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model_to_save.state_dict(), os.path.join(save_path, WEIGHTS_NAME))
        output_config_file = os.path.join(save_path, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    return


def concat_and_truncate(ins):
    max_len = 297
    tokens_a = ins['tokens_a']
    tokens_b = ins['tokens_b']
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    return tokens


def add_segment_id(ins):
    tokens = ins['tokens']
    seg_id = 0
    segment_ids = []
    for token in tokens:
        segment_ids.append(seg_id)
        if token == '[SEP]':
            seg_id = 1
    return segment_ids


def text2feature(dataset, tokenizer, task, is_test=False):
    '''
    把`Dataset`中的文本转化为特征
    :param dataset:
    :param tokenizer:
    :param task:
    :param is_test:
    :return:
    '''
    # tokenize
    dataset.apply(lambda x: tokenizer.tokenize(x['sentence1']), new_field_name='tokens_a')
    dataset.apply(lambda x: tokenizer.tokenize(x['sentence2']), new_field_name='tokens_b')

    # concat and truncate
    dataset.apply(concat_and_truncate, new_field_name='tokens')

    # add segment id
    dataset.apply(add_segment_id, new_field_name='segment_ids')

    # convert tokens to ids
    dataset.apply(lambda x: tokenizer.convert_tokens_to_ids(x['tokens']), new_field_name='input_ids')

    dataset.set_input('input_ids', 'segment_ids')

    if is_test is False:
        # convert label to ids
        if task == 'MNLI':
            label_list = ['neutral', 'entailment', 'contradiction']
        elif task == 'QNLI':
            label_list = ['entailment', 'not_entailment']
        elif task == 'RTE':
            label_list = ['entailment', 'not_entailment']
        elif task == 'WNLI':
            label_list = ['0', '1']
        else:
            raise ValueError('unknown task!')

        label_map = {label: i for i, label in enumerate(label_list)}
        dataset.apply(lambda x: label_map[x['label']], new_field_name='label_id')

        dataset.set_target('label_id')

    return dataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-task', type=str, default='MNLI')
    parser.add_argument('-do_train', type=int, default=1)
    parser.add_argument('-max_len', type=int, default=300)
    parser.add_argument('-n_epoch', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=50)
    parser.add_argument('-gpu', type=str, default='0,1')
    parser.add_argument('-n_gpu', type=int, default=2)
    parser.add_argument('-accumulation_steps', type=int, default=1)
    parser.add_argument('-print_every', type=int, default=100)
	parser.add_argument('-select_epoch', type=int, default=1)
    parser.add_argument('-warmup_proportion', type=float, default=0.1)
    parser.add_argument('-learning_rate', type=float, default=5e-5)
    parser.add_argument('-n_class', type=int, default=3)
    parser.add_argument('-model_descript', type=str,
                        default='MNLI')
    parser.add_argument('-log_dir', type=str,
                        default='/remote-home/txsun/fnlp/watchboard/glue')
    parser.add_argument('-save_path', type=str,
                        default='saved_models/')

    args = parser.parse_args()

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # File Handler
    logger_path = os.path.join('logs', args.task)
    file_handler = logging.FileHandler(logger_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                       datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    logger.info('=============== preparing BERT ===============')
    logger.info('loading vocabulary...')
    bert_path = '/remote-home/txsun/fnlp/bert/BERT_English_uncased_L-12_H-768_A_12'
    vocab_path = 'vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, vocab_path))
    logger.info('done!')
    logger.info('loading pre-trained weights...')
    model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=args.n_class)
    model = model.cuda()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    logger.info('done!')

    logger.info('=========== preparing data: [{}] ==========='.format(args.task))
    data_file = open('data/' + args.task + '.pkl', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    bsz = args.batch_size // args.accumulation_steps

    logger.info('some examples:')
    if args.task == 'MNLI':
        train_ds = text2feature(data['train'], tokenizer, args.task)
        train_dataloader = Batch(train_ds, bsz, sampler=RandomSampler())

        dev_matched_ds = text2feature(data['dev_matched'], tokenizer, args.task)
        dev_matched_dataloader = Batch(dev_matched_ds, bsz, sampler=SequentialSampler())

        dev_mismatched_ds = text2feature(data['dev_mismatched'], tokenizer, args.task)
        dev_mismatched_dataloader = Batch(dev_mismatched_ds, bsz, sampler=SequentialSampler())

        dev_dataloader = [dev_matched_dataloader, dev_mismatched_dataloader]

        test_matched_ds = text2feature(data['test_matched'], tokenizer, args.task, True)
        test_matched_dataloader = Batch(test_matched_ds, bsz, sampler=SequentialSampler())

        test_mismatched_ds = text2feature(data['test_mismatched'], tokenizer, args.task, True)
        test_mismatched_dataloader = Batch(test_mismatched_ds, bsz, sampler=SequentialSampler())

        test_dataloader = [test_matched_dataloader, test_mismatched_dataloader]

        logger.info(train_ds[0])
        logger.info(dev_matched_ds[0])
        logger.info(dev_mismatched_ds[0])
        logger.info(test_matched_ds[0])
        logger.info(test_mismatched_ds[0])
    else:
        train_ds = text2feature(data['train'], tokenizer, args.task)
        train_dataloader = Batch(train_ds, bsz, sampler=RandomSampler())

        dev_ds = text2feature(data['dev'], tokenizer, args.task)
        dev_dataloader = Batch(dev_ds, bsz, sampler=SequentialSampler())

        test_ds = text2feature(data['test'], tokenizer, args.task, True)
        test_dataloader = Batch(test_ds, bsz, sampler=SequentialSampler())

        logger.info(train_ds[0])
        logger.info(dev_ds[0])
        logger.info(test_ds[0])

    logger.info('done!')

    if args.do_train:
        logger.info('================= Training =================')
        num_train_optimization_steps = int(
            len(train_ds) / bsz * args.n_epoch
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        train(model, train_dataloader, dev_dataloader, optimizer, args)
    else:
        logger.info('================= Testing =================')
		load_model_path = os.path.join('saved_models', args.task, args.select_epoch)
		output_config_file = os.path.join(load_model_path, WEIGHTS_NAME)
        output_model_file = os.path.join(load_model_path, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=args.n_class)
        model.load_state_dict(torch.load(output_model_file))

    return


if __name__ == '__main__':
    main()
