import torch
from torch.utils.data import Dataset

__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_feats, audio_feats): 
        
        self.label_ids = label_ids
        self.text_feats = text_feats
        self.video_feats = video_feats
        self.audio_feats = audio_feats

        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_feats[index]),
            'video_feats': torch.tensor(self.video_feats['feats'][index]),
            'audio_feats': torch.tensor(self.audio_feats['feats'][index]),
        }

        return sample