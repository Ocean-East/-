import time
import torch
import os
import random
import numpy as np
from torch.autograd import Variable

from config_parser import args
from lib.loss import SimpleLossCompute
from utils import greedy_decode  # 假设 greedy_decode 已在 utils.py

# 导入 NLTK 的 corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# train_log 函数 (保持您现有的，但请确认它不会在每条消息后自动加两个换行符)
# 我将假设 train_log 期望接收完整的、已包含所需换行的字符串
def train_log(data_to_log, filename_timestamp_str, log_directory="log_train"):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    file_path = os.path.join(log_directory, f'train_log-{filename_timestamp_str}.txt')
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data_to_log)
        # 如果 validate_and_log_random_sample 传递的字符串末尾已有换行，这里就不需要再加 file.write('\n')


# run_epoch 函数 (保持您现有的)
def run_epoch(data, model, loss_compute, epoch, train_log_ts=None):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0  # 新增：记录正确的token数
    tokens = 0

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        
        # 计算accuracy
        pred = out.argmax(dim=-1)  # 获取预测结果
        non_pad_mask = (batch.trg_y != args.PAD)  # 创建非padding位置的mask
        correct = (pred == batch.trg_y).masked_fill(~non_pad_mask, False).sum().item()  # 计算正确的token数
        total_correct += correct

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if train_log_ts and i % 50 == 1:  # 只在训练时记录 batch 日志
            elapsed = time.time() - start
            current_loss_per_token = loss / batch.ntokens
            current_accuracy = correct / batch.ntokens  # 计算当前batch的accuracy
            tokens_per_sec = tokens / elapsed
            if epoch < 0:
                print(f'>>>>> Train Epoch {epoch}')

            log_message = "Epoch %d Batch: %d Loss: %f Accuracy: %.4f Tokens per Sec: %fs" % (
                epoch, i - 1, current_loss_per_token, current_accuracy, tokens_per_sec
            )
            print(log_message)
            train_log(log_message + "\n", train_log_ts)
            start = time.time()
            tokens = 0
    
    # 计算整个epoch的平均accuracy
    avg_accuracy = total_correct / total_tokens
    return total_loss / total_tokens, avg_accuracy  # 返回loss和accuracy


def validate_and_log_random_sample(data_handler, model, epoch, train_log_filename_ts):
    """
    从验证集中随机抽取一个样本进行翻译，计算BLEU分数，并记录到日志。
    输出不包含 BOS/EOS 标记。
    """
    model.eval()

    if not hasattr(data_handler, 'dev_en') or not data_handler.dev_en or \
            not hasattr(data_handler, 'dev_cn') or not data_handler.dev_cn:
        msg = "Validation data (dev_en or dev_cn) not found or empty. Skipping random sample validation."
        print(msg)
        train_log(msg + "\n", train_log_filename_ts)
        model.train()
        return

    num_dev_samples = len(data_handler.dev_en)
    if num_dev_samples == 0:
        msg = "No samples in dev_en to validate."
        print(msg)
        train_log(msg + "\n", train_log_filename_ts)
        model.train()
        return

    random_idx = random.randint(0, num_dev_samples - 1)

    src_indices_with_bos_eos = data_handler.dev_en[random_idx]
    ref_indices_with_bos_eos = data_handler.dev_cn[random_idx]

    # 1. 转换源句文本 (不含BOS/EOS) - 用于显示
    raw_src_tokens_display = [data_handler.en_index_dict.get(w, "UNK") for w in src_indices_with_bos_eos]
    if raw_src_tokens_display and raw_src_tokens_display[0] == "BOS":
        raw_src_tokens_display = raw_src_tokens_display[1:]
    if raw_src_tokens_display and raw_src_tokens_display[-1] == "EOS":
        raw_src_tokens_display = raw_src_tokens_display[:-1]
    src_text_line_display = " ".join(raw_src_tokens_display)

    # 2. 转换参考译文文本 (不含BOS/EOS) - 用于显示和BLEU计算
    raw_ref_tokens_for_bleu_and_display = [data_handler.cn_index_dict.get(w, "UNK") for w in ref_indices_with_bos_eos]
    if raw_ref_tokens_for_bleu_and_display and raw_ref_tokens_for_bleu_and_display[0] == "BOS":
        raw_ref_tokens_for_bleu_and_display = raw_ref_tokens_for_bleu_and_display[1:]
    if raw_ref_tokens_for_bleu_and_display and raw_ref_tokens_for_bleu_and_display[-1] == "EOS":
        raw_ref_tokens_for_bleu_and_display = raw_ref_tokens_for_bleu_and_display[:-1]
    ref_text_line_display = " ".join(raw_ref_tokens_for_bleu_and_display)

    # 准备BLEU计算所需的参考句（列表的列表的词列表）
    # corpus_bleu 需要 list_of_references = [[ref1_tok, ref2_tok, ...], ... ]
    # 对于单个句子，它是 list_of_references = [[[ref_tok_sent1]], [[ref_tok_sent2]], ...]
    # 这里我们只有一个参考翻译，所以是 [[[word1, word2, ...]]]
    list_of_refs_for_bleu = [[raw_ref_tokens_for_bleu_and_display]]

    # 3. 执行翻译
    src_tensor = torch.from_numpy(np.array(src_indices_with_bos_eos)).long().to(args.device)
    src_tensor = src_tensor.unsqueeze(0)
    src_mask = (src_tensor != args.PAD).unsqueeze(-2)

    with torch.no_grad():
        translation_output_indices = greedy_decode(
            model,
            src_tensor,
            src_mask,
            max_len=args.max_length,
            start_symbol=data_handler.cn_word_dict["BOS"],
            device=args.device
        )

    # 4. 转换模型翻译结果文本 (不含BOS/EOS) - 用于显示和BLEU计算
    model_translation_tokens_for_bleu_and_display = []
    for j in range(1, translation_output_indices.size(1)):
        sym_idx = translation_output_indices[0, j].item()
        sym = data_handler.cn_index_dict.get(sym_idx, "UNK")
        if sym == 'EOS':
            break
        model_translation_tokens_for_bleu_and_display.append(sym)
    model_translation_line_display = " ".join(model_translation_tokens_for_bleu_and_display)

    # 准备BLEU计算所需的候选句（词列表的列表）
    # hypotheses = [[hyp1_tok], [hyp2_tok], ...]
    # 对于单个句子，它是 [[word1, word2, ...]]
    hypotheses_for_bleu = [model_translation_tokens_for_bleu_and_display]

    # 5. 计算BLEU分数
    # 定义权重，显式定义，便于修改
    weights_1_gram = (1, 0, 0, 0)
    weights_2_gram = (0, 1, 0, 0)
    weights_3_gram = (0, 0, 1, 0)
    weights_4_gram = (0, 0, 0, 1)

    # 使用 SmoothingFunction 来避免当某个n-gram完全不匹配时分数为0的情况
    chencherry = SmoothingFunction()

    bleu_1 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_1_gram,
                         smoothing_function=chencherry.method1)
    bleu_2 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_2_gram,
                         smoothing_function=chencherry.method1)
    bleu_3 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_3_gram,
                         smoothing_function=chencherry.method1)
    bleu_4 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_4_gram,
                         smoothing_function=chencherry.method1)

    bleu_scores_line = f"bleu : 1_gram = {bleu_1:.4f}  2_gram = {bleu_2:.4f}  3_gram = {bleu_3:.4f}  4_gram = {bleu_4:.4f}"

    # 6. 准备日志条目和打印
    output_header = f"\n--- Epoch {epoch}: Random Validation Sample ---"
    output_src = f"Source:    {src_text_line_display}"
    output_ref = f"Reference: {ref_text_line_display}"
    output_trans = f"Translation: {model_translation_line_display}"
    # BLEU分数行将紧跟翻译结果
    output_footer = "----------------------------------------"  # 移除这里的换行

    # 打印到面板
    print(output_header)
    print(output_src)
    print(output_ref)
    print(output_trans)
    print(bleu_scores_line)  # 打印BLEU分数
    print(output_footer)

    # 写入日志文件 (train_log 现在不自动加换行了)
    full_log_message = f"{output_header}\n{output_src}\n{output_ref}\n{output_trans}\n{bleu_scores_line}\n{output_footer}\n"
    train_log(full_log_message, train_log_filename_ts)

    model.train()


def train(data, model, criterion, optimizer):
    current_time_for_log = time.localtime()
    train_log_filename_ts = time.strftime("%Y-%m-%d-%H-%M-%S", current_time_for_log)

    # 修改日志记录的开头部分
    log_header_content = []
    log_header_content.append(f"Training started at: {train_log_filename_ts}")
    log_header_content.append(f"Model Parameters:")
    log_header_content.append(f"  Layers: {args.layers}")
    log_header_content.append(f"  Head num (h-num): {args.h_num}")
    log_header_content.append(f"  d_model: {args.d_model}")
    log_header_content.append(f"  d_ff: {args.d_ff}")
    log_header_content.append(f"  Max length: {args.max_length}")
    log_header_content.append(f"Training Parameters:")
    log_header_content.append(f"  Epochs: {args.epochs}")
    log_header_content.append(f"  Batch size: {args.batch_size}")
    log_header_content.append(f"  Dropout: {args.dropout}")
    log_header_content.append(f"Optimizer Parameters (NoamOpt):")
    log_header_content.append(
        f"  Warmup steps: {optimizer.warmup if hasattr(optimizer, 'warmup') else 'N/A'}")
    log_header_content.append(
        f"  Factor: {optimizer.factor if hasattr(optimizer, 'factor') else 'N/A'}")
    log_header_content.append("-" * 30)

    train_log("\n".join(log_header_content) + "\n\n", train_log_filename_ts)

    for epoch in range(args.epochs):
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch,
                  train_log_filename_ts)

        model.eval()
        eval_log_message_start = f'\n>>>>> Evaluate Epoch {epoch} Validation Loss and Accuracy <<<<<\n'
        print(eval_log_message_start.strip())
        train_log(eval_log_message_start, train_log_filename_ts)

        loss, accuracy = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch,
                         None)

        eval_log_message = f'<<<<< Evaluate Epoch {epoch} - Loss: {loss:.4f} Accuracy: {accuracy:.4f} >>>>>\n'
        print(eval_log_message.strip())
        train_log(eval_log_message, train_log_filename_ts)

        validate_and_log_random_sample(data, model, epoch, train_log_filename_ts)

    torch.save(model.state_dict(), args.save_file)
    train_log(f"\nTraining finished. Model saved to {args.save_file}\n", train_log_filename_ts)