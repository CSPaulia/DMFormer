import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel

import os
import logging
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode


def run_epoch(data, model, loss_compute, device):
    total_tokens = 0.
    total_loss = 0.

    for _, batch in tqdm(data):
        batch.to(device)
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def mode(data, ignore=0):
    '''
    求 2-d tensor数据的每一列数据的出现次数最多的元素，并忽略元素ignore
    '''
    _, lines = data.shape
    moded_data = []
    for i in range(lines):
        line = data[:, i]
        unique_data, counts = torch.unique(line, return_counts=True)
        ignore_loc = torch.where(unique_data == ignore)[0]
        if torch.numel(ignore_loc) != 0:
            unique_data = torch.cat((unique_data[:ignore_loc], unique_data[ignore_loc+1:]))
            counts = torch.cat((counts[:ignore_loc], counts[ignore_loc+1:]))
        moded_data.append(unique_data[torch.argmax(counts)])
    return torch.tensor(moded_data)


def recover2origin(origin_idx, pred, target, window_size, pad):
    '''
    将经过window划窗切割的数据恢复成原来的样式。
    '''
    unique_idx = torch.unique(origin_idx)
    recovered_preds = []
    recovered_trgs = []
    for idx in unique_idx:
        # 找出属于idx曲子的预测值和真实值
        idx_scope = torch.where(origin_idx == idx)
        idx_pred = pred[idx_scope]
        idx_trg = target[idx_scope]

        # 如果属于idx曲子的预测值和真实值的长度为1，则不需要恢复
        if len(idx_pred) == 1:
            assert len(idx_pred) == len(idx_trg)
            recovered_preds.append(idx_pred[idx_trg != pad])
            recovered_trgs.append(idx_trg[idx_trg != pad])
        # 将经过window划窗切割的数据恢复成原来的样式
        else:
            assert len(idx_pred) == len(idx_trg)
            recover_pred_array = torch.zeros((len(idx_pred), window_size + len(idx_pred) - 1))
            recover_trg_array = torch.zeros((len(idx_trg), window_size + len(idx_trg) - 1))
            for i in range(len(idx_pred)):
                recover_pred_array[i, i:i+window_size] = idx_pred[i]
                recover_trg_array[i, i:i+window_size] = idx_trg[i]
            recovered_pred = mode(recover_pred_array)
            recovered_trg = mode(recover_trg_array)
            recovered_preds.append(recovered_pred)
            recovered_trgs.append(recovered_trg)

    return recovered_preds, recovered_trgs


def evaluate(data, model, device, use_window=False, window_size=None):
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        total_correct = 0
        total_recovered_correct = 0
        total_ntockens = 0
        total_recovered_ntokens = 0

        for origin_idx, batch in tqdm(data):
            origin_idx = origin_idx.to(device)
            batch.to(device)

            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            out = model.module.generator(out)
            pred = torch.argmax(out, dim=-1)

            pred_mask = (batch.trg_y != batch.pad)
            correct = (pred == batch.trg_y) & pred_mask
            total_correct += torch.sum(correct).item()
            total_ntockens += batch.ntokens

            if use_window:
                recovered_preds, recovered_trgs = recover2origin(origin_idx, pred, batch.trg_y, window_size, batch.pad)
                for i in range(len(recovered_preds)):
                    assert len(recovered_preds[i]) == len(recovered_trgs[i])
                    total_recovered_correct += torch.sum(recovered_preds[i] == recovered_trgs[i]).item()
                    total_recovered_ntokens += len(recovered_preds[i])

        acc = total_correct / total_ntockens
        if use_window:
            recovered_acc = total_recovered_correct / total_recovered_ntokens
            return acc, recovered_acc
        else:
            return acc


def train(train_data, dev_data, model, criterion, optimizer, device, writer, use_window=False, window_size=None):
    """训练并保存模型"""
    best_acc = 0
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model,
                               LossCompute(model.module.generator, criterion, optimizer), device)
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        writer.add_scalar('train/loss', train_loss, epoch)
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model,
                             LossCompute(model.module.generator, criterion, None), device)
        if use_window:
            acc, recovered_acc = evaluate(dev_data, model, device, use_window, window_size)
            logging.info('Epoch: {}, Dev loss: {}, Acc: {}, Recovered Acc: {}'.format(epoch, dev_loss, acc, recovered_acc))
            writer.add_scalar('validate/loss', dev_loss, epoch)
            writer.add_scalar('validate/acc', acc, epoch)
            writer.add_scalar('validate/recovered_acc', recovered_acc, epoch)
            if recovered_acc > best_acc:
                best_acc = recovered_acc
                torch.save(model.state_dict(), os.path.join(config.model_path, 'best.pth'))
        else:
            acc = evaluate(dev_data, model, device)
            logging.info('Epoch: {}, Dev loss: {}, Acc: {}'.format(epoch, dev_loss, acc))
            writer.add_scalar('validate/loss', dev_loss, epoch)
            writer.add_scalar('validate/acc', acc, epoch)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(config.model_path, 'best.pth'))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        torch.save(model.state_dict(), os.path.join(config.model_path, f'epoch{epoch}.pth'))
    torch.save(model.state_dict(), os.path.join(config.model_path, 'last.pth'))

class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data.item() * norm.float()


def test(data, model, criterion, device, use_window=False, window_size=None):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(os.path.join(config.model_path, 'best.pth')))
        model.eval()

        if use_window:
            acc, recovered_acc = evaluate(data, model, device, use_window, window_size)
            logging.info('Acc: {}, Recovered Acc: {}'.format(acc, recovered_acc))
        else:
            acc = evaluate(data, model, device)
            logging.info('Acc: {}'.format(acc))