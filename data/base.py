import os
import logging
import csv
from torch.utils.data import DataLoader

from .__init__ import benchmarks
from .text_pre import get_t_data
from .video_pre import get_v_data
from .audio_pre import get_a_data
from .mm_pre import MMDataset

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)

        bm = benchmarks[args.dataset]
        
        self.label_list = bm["intent_labels"]
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))  
        args.num_labels = len(self.label_list)

        max_seq_lengths = bm['max_seq_lengths']
        args.text_seq_len = max_seq_lengths['text']
        args.video_seq_len = max_seq_lengths['video']
        args.audio_seq_len = max_seq_lengths['audio']

        self.mm_data = get_dataset(args, self.label_list) 

        
def get_dataset(args, label_list):
    
    data_path = os.path.join(args.data_path, args.dataset)
    text_data_path = data_path
    video_feats_path = os.path.join(data_path, args.video_data_path, args.video_feats_path)
    audio_feats_path = os.path.join(data_path, args.audio_data_path, args.audio_feats_path)

    train_data_index, train_label_ids = get_indexes_annotations(args, label_list, os.path.join(data_path, 'train.tsv'))
    dev_data_index, dev_label_ids = get_indexes_annotations(args, label_list, os.path.join(data_path, 'dev.tsv'))
    test_data_index, test_label_ids = get_indexes_annotations(args, label_list, os.path.join(data_path, 'test.tsv'))
    args.num_train_examples = len(train_data_index)    

    data_args = {
        'data_path': data_path,
        'text_data_path': text_data_path,
        'video_feats_path': video_feats_path,
        'audio_feats_path': audio_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    
    text_data = get_t_data(args, data_args)
    video_feats_data = get_v_data(args, data_args)
    audio_feats_data = get_a_data(args, data_args)
    
    train_data = MMDataset(train_label_ids, text_data['train'], video_feats_data['train'], audio_feats_data['train'])
    dev_data = MMDataset(dev_label_ids, text_data['dev'], video_feats_data['dev'], audio_feats_data['dev']) 
    test_data = MMDataset(test_label_ids, text_data['test'], video_feats_data['test'], audio_feats_data['test'])

    mm_data = {'train': train_data, 'dev': dev_data, 'test': test_data}     
    
    return mm_data

def get_dataloader(args, mm_data):

    train_dataloader = DataLoader(mm_data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
    dev_dataloader = DataLoader(mm_data['dev'], shuffle=False, batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
    test_dataloader = DataLoader(mm_data['test'], shuffle=False, batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)

    dataloader = {
        'train': train_dataloader,
        'dev': dev_dataloader,
        'test': test_dataloader
    }  
    return dataloader

def get_indexes_annotations(args, label_list, read_file_path):

        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        with open(read_file_path, 'r') as f:

            data = csv.reader(f, delimiter="\t")
            indexes = []
            label_ids = []

            for i, line in enumerate(data):
                if i == 0:
                    continue
                
                if args.dataset in ['MIntRec']:
                    index = '_'.join([line[0], line[1], line[2]])                
                    label_id = label_map[line[4]]          
                
                elif args.dataset in ['MIntRec2.0']:
                    index ='_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])   
                    label_id = label_map[line[3]]

                elif args.dataset in ['MELD-DA']:
                    index = '_'.join([line[0], line[1]])
                    label_id = label_map[line[3]]
                
                elif args.dataset in ['IEMOCAP-DA']:
                    index = line[0]
                    label_id = label_map[line[2]]
                
                indexes.append(index)
                label_ids.append(label_id)

        return indexes, label_ids