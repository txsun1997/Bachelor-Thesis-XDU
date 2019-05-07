import sys
import os
import pickle
from fastNLP import Vocabulary
from fastNLP import DataSet

root_path = '/remote-home/txsun/data/glue_data'

# QNLI
print('processing QNLI...')
dataset = 'QNLI'
data_path = os.path.join(root_path, dataset)

## Train
print('reading train file...')
train_file = os.path.join(data_path, 'train.tsv')
train_ds = DataSet.read_csv(train_file, sep='\t')
train_ds.delete_field('index')
train_ds.rename_field('question', 'sentence1')
train_ds.rename_field('sentence', 'sentence2')
print(train_ds[0])
print(len(train_ds))

## Dev
print('reading dev file...')
dev_file = os.path.join(data_path, 'dev.tsv')
dev_ds = DataSet.read_csv(dev_file, sep='\t')
dev_ds.delete_field('index')
dev_ds.rename_field('question', 'sentence1')
dev_ds.rename_field('sentence', 'sentence2')
print(dev_ds[0])
print(len(dev_ds))

## Test
print('reading test file...')
test_file = os.path.join(data_path, 'test.tsv')
test_ds = DataSet.read_csv(test_file, sep='\t')
test_ds.delete_field('index')
test_ds.rename_field('question', 'sentence1')
test_ds.rename_field('sentence', 'sentence2')
print(test_ds[0])
print(len(test_ds))

# Save data
print('dumping data...')
data = {
    'train': train_ds,
    'dev': dev_ds,
    'test': test_ds
}
save_file = open('data/QNLI.pkl', 'wb')
pickle.dump(data, save_file)
save_file.close()
print('data saved.')