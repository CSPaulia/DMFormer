import utils
import yaml
import config
import logging
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from train import train, test, translate
from data_loader import DMDataset
from utils import english_tokenizer_load
from model import make_model

def initialize_distributed_training():
    master_addr = config.master_addr
    master_port = config.master_port
    
    rank = config.rank
    world_size = config.world_size

    # 初始化分布式训练环境
    dist.init_process_group(
        backend='nccl',  # 使用 nccl 作为后端
        init_method=f'tcp://{master_addr}:{master_port}',  # 主节点地址和端口
        rank=rank,  # 当前进程的 rank
        world_size=world_size  # 总进程数
    )

def run(args):
    initialize_distributed_training()

    with open(config.label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map']

    utils.set_logger(config.log_path)

    train_dataset = DMDataset(config.train_data_path, config.window_size, learning_map, config.use_window)
    dev_dataset = DMDataset(config.dev_data_path, config.window_size, learning_map, config.use_window)
    test_dataset = DMDataset(config.test_data_path, config.window_size, learning_map, config.use_window)

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)
    test_sampler = DistributedSampler(test_dataset)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn, sampler=train_sampler)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn, sampler=dev_sampler)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn, sampler=test_sampler)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(config.src_embedding_size, config.src_embedding_size, config.class_num, 
                       config.n_layers, config.d_model, config.d_ff, config.n_heads, config.dropout)
    device = torch.device(f"cuda:{args.local_rank}")
    model.to(device)
    model = DistributedDataParallel(model)

    # 训练
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataloader, dev_dataloader, model, criterion, optimizer, device, config.use_window, config.window_size)
    test(test_dataloader, model, criterion)


def one_sentence_translate(sent, beam_search=True):
    # 初始化模型
    model = make_model(config.src_embedding_size, config.src_embedding_size, config.class_num, 
                       config.n_layers, config.d_model, config.d_ff, config.n_heads, config.dropout)
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """单句翻译示例"""
    sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
           "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
           "to childless workers."
    # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
    one_sentence_translate(sent, beam_search=True)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--learning_map", type=str, default='/root/GenGM/utils/DrumMusic.yaml', help='learning map')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    args = set_args()
    run(args)
    # translate_example()
