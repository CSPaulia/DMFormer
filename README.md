Language: 简体中文

# DMFormer

基于transformer的西安鼓乐韵曲模型。

## Data

已韵的西安鼓乐

## Data Process

## Model

采用Harvard开源的 [transformer-pytorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html) ，中文说明可参考 [传送门](https://zhuanlan.zhihu.com/p/144825330) 。

## Requirements

This repo was tested on Python 3.8 and PyTorch 1.10.1. The main requirements are:

- tqdm
- pytorch >= 1.5.1

To get the environment settled quickly, run:

```
pip install -r requirements.txt
```

## Usage

模型参数在`config.py`中设置。

- 由于transformer显存要求，支持MultiGPU，需要设置`config.py`中的`gpus`列。

如要运行模型，可在命令行输入：

``` shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE main.py
# nproc_per_node: 这个参数是指你使用这台服务器上面的几张显卡
```

## Results

| Model |  Best Acc  | Best Recovered Acc |
| :---: |  :------:  | :----------------: |
|   1   |   24.07    |        24.03       |
|   2   |     -      |          -         |
|   3   |     -      |          -         |

## Pretrained Model

训练好的 Model 1 模型（当前最优模型）可以在如下链接直接下载😊：

链接: https://pan.baidu.com/s/1RKC-HV_UmXHq-sy1-yZd2Q  密码: g9wl

## Mention

The codes released in this reposity are only tested successfully with **Linux**.

## Todo List

- [ ] windowed_data函数暂不支持window_size大于乐曲长度，需要更新
- [ ] 使用Greedy Decoder和Beam Search（不确定适用性）
- [ ] 暂时只关注了有音高的音符，后续加入空拍的编码
- [ ] 从输出预测中恢复出midi文件
- [ ] 修改模型使模型关注小节/乐句内容
- [ ] 清洗数据集
- [ ] 预训练模型

