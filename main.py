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
from torch.utils.tensorboard import SummaryWriter

from train import train, test
from data_loader import DMDataset
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
    # 初始化分布式训练环境
    initialize_distributed_training()

    # 打开Drum Music的yaml文件，包含时值与标签的对应关系
    with open(config.label_mapping, 'r') as stream:
        drummusicyaml = yaml.safe_load(stream)
    learning_map = drummusicyaml['learning_map']

    # 设置logger
    utils.set_logger(config.log_path)

    # 设置tensorboard
    writer = SummaryWriter(config.event_path)

    # 载入训练集、验证集以及测试集
    train_dataset = DMDataset(config.train_data_path, config.window_size, learning_map, config.use_window)
    dev_dataset = DMDataset(config.dev_data_path, config.window_size, learning_map, config.use_window)
    test_dataset = DMDataset(config.test_data_path, config.window_size, learning_map, config.use_window)

    # 对数据集进行分布式采样，用于分布式训练（多GPU训练）
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
    # train(train_dataloader, dev_dataloader, model, criterion, optimizer, device, writer, config.use_window, config.window_size)
    test(test_dataloader, model, criterion, device, config.use_window, config.window_size)

    writer.close()

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--learning_map", type=str, default='/root/GenGM/utils/DrumMusic.yaml', help='learning map')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpus)[1:-1]

    import warnings
    warnings.filterwarnings('ignore')

    args = set_args()

    run(args)
