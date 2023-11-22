import torch
import os
import mido
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            self.trg = torch.cat([torch.zeros(self.trg.size(0), 1), self.trg], dim=1).to(torch.int64)
            # decoder训练时应预测输出的target结果
            self.trg_y = trg
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()
        self.pad = pad

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    def to(self, device):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        self.trg = self.trg.to(device)
        self.trg_y = self.trg_y.to(device)
        self.trg_mask = self.trg_mask.to(device)
        self.ntokens = self.ntokens.to(device)
    
    
class midi_note():
    def __init__(self, type, note, velocity, time):
        self.type = type
        self.note = note
        self.velocity = velocity
        self.time = time

class DMDataset(Dataset):
    def __init__(self, data_path, window_size, learning_map, use_window=False, pad=0):
        self.PAD = pad
        self.window_size = window_size
        self.learning_map = learning_map
        self.use_window = use_window
        self.duration_class = np.array(list(learning_map.keys()))
        self.pitch_values, self.duration_idx_values = self.get_dataset_pitch_and_duration(data_path)
        if use_window:
            self.origin_idx, self.pitch_values, self.duration_idx_values = self.windowed_data(self.pitch_values, self.duration_idx_values, window_size)
        else:
            self.origin_idx = np.arange(len(self.pitch_values))


    def duration2class(self, duration):
        '''
        将时长转换成标签，时长和标签的对应关系保存在DrumMusic.yaml中
        '''
        class_idx = np.argmin(np.abs(self.duration_class - duration))
        class_idx = self.learning_map[self.duration_class[class_idx]]
        return class_idx

    
    def get_pitch_and_duration(self, midi_path):
        '''
        获得midi文件，并提取midi文件中每一个音符的音高和时长
        '''
        # 存储音高和时长的列表
        pitch_values = []
        duration_idx_values = []

        # 读取 MIDI 文件
        midi_file = mido.MidiFile(midi_path)

        for track in midi_file.tracks:
            notes = []
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    one_note = midi_note(msg.type, msg.note, msg.velocity, msg.time)
                    notes.append(one_note)

            for i in range(len(notes)):
                # 提取音符的音高和时长信息
                # notes[i].type == 'note_on' and notes[i].velocity > 0 是一个音符的开始
                if notes[i].type == 'note_on' and notes[i].velocity > 0:
                    duration = 0
                    for j in range(i+1, len(notes)):
                        duration += notes[j].time
                        # notes[j].type == 'note_off' or (notes[j].type == 'note_on' and notes[j].velocity == 0) 是一个音符的结束
                        # 还需要判断结束和开始的音高是否相同
                        if (notes[j].type == 'note_off' or (notes[j].type == 'note_on' and notes[j].velocity == 0)) and notes[i].note == notes[j].note:
                            break
                    if duration > 10000:
                        break
                    duration_idx_values.append(self.duration2class(duration))
                    pitch_values.append(notes[i].note)

        return pitch_values, duration_idx_values
    
    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset_pitch_and_duration(self, dataset_path):
        '''
        获得dataset_path下的所有midi文件，提取midi文件中每一个音符的音高和时长并保存
        '''
        all_pitchs = []
        all_duration_idxs = []
        for midi_name in os.listdir(dataset_path):
            midi_path = os.path.join(dataset_path, midi_name)
            pitchs, duration_idxs = self.get_pitch_and_duration(midi_path)
            all_pitchs.append(pitchs)
            all_duration_idxs.append(duration_idxs)
        
        sorted_index = self.len_argsort(all_pitchs)
        all_pitchs = [all_pitchs[i] for i in sorted_index]
        all_duration_idxs = [all_duration_idxs[i] for i in sorted_index]

        return all_pitchs, all_duration_idxs
    
    def windowed_data(self, pitchs, durations, window_size):
        '''
        将midi数据按照window大小进行切割。
        示例：
        >>> pitchs = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 12, 13, 14]]
        >>> durations = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 12, 13, 14]]
        >>> window_size = 4
        >>> origin_idx, windowed_pitchs, windowed_durations = windowed_data(pitchs, durations, window_size)
        >>> origin_idx
        [0, 0, 0, 0, 0, 0, 1]
        >>> windowed_pitchs
        [
            [1, 2, 3, 4]
            [2, 3, 4, 5]
            [3, 4, 5, 6]
            [4, 5, 6, 7]
            [5, 6, 7, 8]
            [6, 7, 8, 9]
            [11, 12, 13, 14]
        ]
        '''
        pitch_values = pitchs
        duration_idx_values = durations

        origin_idx = []
        windowed_pitchs = []
        windowed_durations = []
        for idx, (pitchs, durations) in enumerate(zip(pitch_values, duration_idx_values)):
            assert len(pitchs) == len(durations)
            for j in range(len(pitchs) - window_size + 1):
                origin_idx.append(idx)
                windowed_pitchs.append(pitchs[j: j+window_size])
                windowed_durations.append(durations[j: j+window_size])

        return origin_idx, windowed_pitchs, windowed_durations
    
    def __getitem__(self, idx):
        origin_idx = self.origin_idx[idx]
        pitchs = self.pitch_values[idx]
        durations = self.duration_idx_values[idx]
        return [origin_idx, pitchs, durations]

    def __len__(self):
        return len(self.pitch_values)

    def collate_fn(self, batch):
        origin_idx = [x[0] for x in batch]
        src = [x[1] for x in batch]
        tgt = [x[2] for x in batch]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt],
                                    batch_first=True, padding_value=self.PAD)

        return torch.LongTensor(origin_idx), Batch(batch_input, batch_target, self.PAD)