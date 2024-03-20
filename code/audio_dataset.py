import os
import torch
from torch.utils.data import Dataset, DataLoader
from data_process import extract_hpss_features_sg

class AudioDataset(Dataset):
    def __init__(self, data_dir, max_len, window_length, window_shift, use_stft):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.max_len = max_len
        self.window_shift = window_shift
        self.window_length = window_length
        self.use_stft = use_stft

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = os.path.join(self.data_dir, self.file_list[idx])
        wav_data = extract_hpss_features_sg(filename, max_length=self.max_len, window_length=self.window_length, window_shift=self.window_shift, use_stft=self.use_stft)
        wav_data = torch.tensor(wav_data)
        wav_data = wav_data.unsqueeze(0)

        # Parse label from filename (filename format: id1_filename.wav)
        label = self.file_list[idx].split('_')[0][2:]  # Extract label from filename
        label = torch.tensor([int(label)])

        return wav_data, label