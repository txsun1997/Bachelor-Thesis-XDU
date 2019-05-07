import sys
import os
import pickle
from fastNLP import Vocabulary
from fastNLP import DataSet

root_path = '/remote-home/txsun/data/glue_data'

# MNLI
print('processing MNLI...')
dataset = 'MNLI'
data_path = os.path.join(root_path, dataset)
delete_test_field_lst = ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                         'sentence1_parse', 'sentence2_parse']
delete_train_field_lst = ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                          'sentence1_parse', 'sentence2_parse', 'label1']
delete_dev_field_lst = delete_train_field_lst + ['label2', 'label3', 'label4', 'label5']

## Train
print('reading train file...')
train_file = os.path.join(data_path, 'train.tsv')
train_ds = DataSet.read_csv(train_file, sep='\t')

for field_name in delete_train_field_lst:
    train_ds.delete_field(field_name)

train_ds.rename_field('gold_label', 'label')
print(train_ds[0])
print(len(train_ds))

## Dev_matched
print('reading dev_matched file...')
dev_matched_file = os.path.join(data_path, 'dev_matched.tsv')
dev_matched_ds = DataSet.read_csv(dev_matched_file, sep='\t')

for field_name in delete_dev_field_lst:
    dev_matched_ds.delete_field(field_name)

dev_matched_ds.rename_field('gold_label', 'label')
print(dev_matched_ds[0])
print(len(dev_matched_ds))

## Dev_mismatched
print('reading dev_mismatched file...')
dev_mismatched_file = os.path.join(data_path, 'dev_mismatched.tsv')
dev_mismatched_ds = DataSet.read_csv(dev_mismatched_file, sep='\t')

for field_name in delete_dev_field_lst:
    dev_mismatched_ds.delete_field(field_name)

dev_mismatched_ds.rename_field('gold_label', 'label')
print(dev_mismatched_ds[0])
print(len(dev_mismatched_ds))

## Test_matched
print('reading test_matched file...')
test_matched_file = os.path.join(data_path, 'test_matched.tsv')
test_matched_ds = DataSet.read_csv(test_matched_file, sep='\t')

for field_name in delete_test_field_lst:
    test_matched_ds.delete_field(field_name)

print(test_matched_ds[0])
print(len(test_matched_ds))

## Test_mismatched
print('reading test_mismatched file...')
test_mismatched_file = os.path.join(data_path, 'test_mismatched.tsv')
test_mismatched_ds = DataSet.read_csv(test_mismatched_file, sep='\t')

for field_name in delete_test_field_lst:
    test_mismatched_ds.delete_field(field_name)

print(test_mismatched_ds[0])
print(len(test_mismatched_ds))

# Save data
print('dumping data...')
data = {
    'train': train_ds,
    'dev_matched': dev_matched_ds,
    'dev_mismatched': dev_mismatched_ds,
    'test_matched': test_matched_ds,
    'test_mismatched': test_mismatched_ds
}
save_file = open('data/MNLI.pkl', 'wb')
pickle.dump(data, save_file)
save_file.close()
print('data saved.')