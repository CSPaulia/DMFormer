import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import logging
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import chinese_tokenizer_load


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
    unique_idx = torch.unique(origin_idx)
    recovered_preds = []
    recovered_trgs = []
    for idx in unique_idx:
        idx_scope = torch.where(origin_idx == idx)
        idx_pred = pred[idx_scope]
        idx_trg = target[idx_scope]
        if len(idx_pred) == 1:
            assert len(idx_pred) == len(idx_trg)
            recovered_preds.append(idx_pred[idx_trg != pad])
            recovered_trgs.append(idx_trg[idx_trg != pad])
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


def evaluate(data, model, device, use_window=False, window_size=None, mode='dev', use_beam=True):
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
            
    # if mode == 'test':

    # return float(bleu.score)


def train(train_data, dev_data, model, criterion, optimizer, device, use_window=False, window_size=None):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model,
                               LossCompute(model.module.generator, criterion, optimizer), device)
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model,
                             LossCompute(model.module.generator, criterion, None), device)
        if use_window:
            acc, recovered_acc = evaluate(dev_data, model, device, use_window, window_size)
            logging.info('Epoch: {}, Dev loss: {}, Acc: {}, Recovered Acc: {}'.format(epoch, dev_loss, acc, recovered_acc))
        else:
            acc = evaluate(dev_data, model, device)
            logging.info('Epoch: {}, Dev loss: {}, Acc: {}'.format(epoch, dev_loss, acc))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        torch.save(model.state_dict(), os.path.join(config.model_path, f'epoch{epoch}.pth'))

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


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        print(translation[0])

if __name__ == '__main__':
    origin_idx = torch.tensor([0,0,0,1,2,2])
    pred = torch.tensor([
        [1,2,2],
        [2,3,1],
        [2,1,3],
        [1,2,0],
        [2,3,2],
        [2,2,1]
    ])
    trg = torch.tensor([
        [1,2,2],
        [2,2,1],
        [2,1,3],
        [1,2,0],
        [2,3,2],
        [3,2,1]
    ])
    window_size = 3
    pad = 0
    rp, rt = recover2origin(origin_idx, pred, trg, window_size, pad)
    print(rp, rt)