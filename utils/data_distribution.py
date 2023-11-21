import mido
import os
import numpy as np
import matplotlib.pyplot as plt

duration_class = np.array([0, 4, 9, 22, 45, 60, 90, 120, 150, 180, 210, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 1920, 2400, 2880, 3840, 4800])

class midi_note():
    def __init__(self, type, note, velocity, time):
        self.type = type
        self.note = note
        self.velocity = velocity
        self.time = time

def duration2class(duration):
    class_idx = np.argmin(np.abs(duration_class - duration))
    return duration_class[class_idx]

def get_pitch_and_duration(midi_path):
    # 存储音高和时长的列表
    pitch_values = []
    duration_values = []

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
            if notes[i].type == 'note_on' and notes[i].velocity > 0:
                duration = 0
                for j in range(i+1, len(notes)):
                    duration += notes[j].time
                    if (notes[j].type == 'note_off' or (notes[j].type == 'note_on' and notes[j].velocity == 0)) and notes[i].note == notes[j].note:
                        break
                if duration > 10000:
                    break
                duration_values.append(duration2class(duration))
                pitch_values.append(notes[i].note)

    return pitch_values, duration_values

def get_dataset_pitch_and_duration(dataset_path):
    all_pitchs = []
    all_durations = []
    for midi_name in os.listdir(dataset_path):
        midi_path = os.path.join(dataset_path, midi_name)
        pitchs, durations = get_pitch_and_duration(midi_path)
        all_pitchs.append(pitchs)
        all_durations.append(durations)
    
    all_pitchs = np.concatenate(all_pitchs)
    all_durations = np.concatenate(all_durations)

    return all_pitchs, all_durations


def plot_histogram(data, title, xlabel, ylabel):
    plt.hist(data, bins=50, alpha=0.75, color='b', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':
    # 示例用法
    midi_dataset_path = 'C:\\Users\\CSPau\\OneDrive\\文档\\GenGM\\data\\midiAll'  # 替换为你的 MIDI 数据集文件路径
    pitch_values, duration_values= get_dataset_pitch_and_duration(midi_dataset_path)

    print('pitch class', np.unique(pitch_values))
    unique_duration_values, counts = np.unique(duration_values, return_counts=True)
    print('duration class')
    for v, c in zip(unique_duration_values, counts):
        print(v, c)

    # 绘制音高和时长的直方图
    plot_histogram(pitch_values, 'Pitch Distribution', 'Pitch', 'Frequency')
    plot_histogram(duration_values, 'Duration Distribution', 'Duration', 'Frequency')
