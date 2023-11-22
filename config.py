import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_embedding_size = 512
tgt_embedding_size = 512

batch_size = 128
epoch_num = 40
early_stop = 5
lr = 3e-4

class_num = 30
use_window = True
window_size = 60

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 3

data_dir = './data'
train_data_path = '/mnt/data/repos/GenGM/data/midiAll/train'
dev_data_path = '/mnt/data/repos/GenGM/data/midiAll/validate'
test_data_path = '/mnt/data/repos/GenGM/data/midiAll/test'
model_path = './experiment'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'
label_mapping = '/mnt/data/repos/GenGM/utils/DrumMusic.yaml'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]

# 设置分布式训练环境的主节点信息
master_addr = "localhost"
master_port = "1367"
    
# 设置当前进程的 rank 和总进程数
rank = 0  # 当前进程的 rank
world_size = 1  # 总进程数