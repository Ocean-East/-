import copy
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable # 需要导入 Variable


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 处理测试数据
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # triangle upper
    return torch.from_numpy(subsequent_mask) == 0  # TODO 这里的遮挡关系还需要理解，为什么是和0作对比

def greedy_decode(model, src, src_mask, max_len, start_symbol, device): # 添加 device 参数
    """
    Performs greedy decoding.
    model: The transformer model.
    src: Source sentence tensor.
    src_mask: Source sentence mask.
    max_len: Maximum length for the output sentence.
    start_symbol: The start-of-sentence symbol for the target language.
    device: The device to run the computations on (e.g., 'cuda' or 'cpu').
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long().to(device) # 确保在正确设备上
    for i in range(max_len - 1):
        # 确保后续的 mask 也创建在正确的设备上
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data).to(device)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word).long().to(device)], dim=1) # 确保在正确设备上
    return ys
