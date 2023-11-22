import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import config
import tqdm



def compose(data, model, device, use_window=False, window_size=None):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model = DistributedDataParallel(model)
        model.eval()

        for origin_idx, batch in tqdm(data):
            origin_idx = origin_idx.to(device)
            batch.to(device)

            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            out = model.module.generator(out)
            pred = torch.argmax(out, dim=-1)