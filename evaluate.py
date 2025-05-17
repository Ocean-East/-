import torch
import numpy as np
import time
import os

from config_parser import args
from torch.autograd import Variable
# 从 utils.py 导入需要的函数
from utils import subsequent_mask, greedy_decode # 确保 greedy_decode 在 utils.py 中

# 导入 NLTK 的 corpus_bleu 和 SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def log(data_to_log, timestamp_str, log_directory="log_evaluate", is_final_summary=False):
    """
    将评估日志数据写入指定文件。
    :param data_to_log: 要记录的字符串数据。
    :param timestamp_str: 用于日志文件名的格式化时间戳。
    :param log_directory: 日志文件存放的目录名。
    :param is_final_summary: 是否是最终的摘要信息，如果是，则可能需要不同的处理或标记。
    """
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 日志文件名保持 evaluate_log-{timestamp_str}.txt
    file_path = os.path.join(log_directory, f'evaluate_log-{timestamp_str}.txt')
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data_to_log)
        # 每个log调用负责自己的换行逻辑，除非是追加多行块
        # if not data_to_log.endswith('\n'):
        #     file.write('\n')


# greedy_decode 函数应已移至 utils.py
# def greedy_decode(model, src, src_mask, max_len, start_symbol, device): ...


def evaluate(data, model):
    # 在 evaluate 函数开始时获取当前时间并格式化
    eval_start_time = time.localtime()
    timestamp_log_str = time.strftime("%Y-%m-%d-%H-%M-%S", eval_start_time)

    # 初始化用于存储所有句子 BLEU 分数的列表
    all_bleu_1_scores = []
    all_bleu_2_scores = []
    all_bleu_3_scores = []
    all_bleu_4_scores = []

    # 在函数开始处添加新的列表来存储准确率
    all_accuracy_scores = []  # 存储所有句子的准确率

    # 定义BLEU权重，显式定义
    weights_1_gram = (1, 0, 0, 0)
    weights_2_gram = (0, 1, 0, 0)
    weights_3_gram = (0, 0, 1, 0)
    weights_4_gram = (0, 0, 0, 1)
    chencherry = SmoothingFunction() # 平滑函数

    log_header_content = []
    log_header_content.append(f"Evaluation started at: {timestamp_log_str}")
    log_header_content.append(f"Model File: {args.save_file}") # 评估用的模型文件
    log_header_content.append(f"Dataset (dev_file): {args.dev_file}") # 评估用的数据集
    log_header_content.append(f"Model Parameters (from config):")
    log_header_content.append(f"  Layers: {args.layers}")
    log_header_content.append(f"  Head num (h-num): {args.h_num}")
    log_header_content.append(f"  d_model: {args.d_model}")
    log_header_content.append(f"  d_ff: {args.d_ff}")
    log_header_content.append(f"Evaluation Parameters:")
    log_header_content.append(f"  Max length (for greedy_decode): {args.max_length}")
    log_header_content.append(f"  Batch size (if applicable to eval, usually 1 for sentence-by-sentence): {args.batch_size}") # 注意：评估时batch_size行为可能不同
    log_header_content.append("-" * 30)  # 分隔符

    # 将所有头部信息一次性写入，每条占一行，头部信息块后加两个换行
    log("\n".join(log_header_content) + "\n\n", timestamp_log_str)
    # --- 参数信息打印结束 ---


    with torch.no_grad():
        model.eval() # 确保模型在评估模式

        for i in range(len(data.dev_en)):
            log_entry_parts = [] # 用于收集当前样本的所有日志行

            # --- 文本处理，移除BOS/EOS，用于显示和BLEU ---
            # 1. 英文原文 (仅显示，不参与BLEU)
            raw_en_tokens_display = [data.en_index_dict.get(w, "UNK") for w in data.dev_en[i]]
            if raw_en_tokens_display and raw_en_tokens_display[0] == "BOS":
                raw_en_tokens_display = raw_en_tokens_display[1:]
            if raw_en_tokens_display and raw_en_tokens_display[-1] == "EOS":
                raw_en_tokens_display = raw_en_tokens_display[:-1]
            en_sent_display = " ".join(raw_en_tokens_display)
            print("\n" + en_sent_display)
            log_entry_parts.append(en_sent_display)

            # 2. 中文参考译文 (用于显示和BLEU)
            # data.dev_cn[i] 是包含BOS/EOS的索引列表
            ref_indices_with_bos_eos = data.dev_cn[i]
            raw_ref_tokens_for_bleu = [data.cn_index_dict.get(w, "UNK") for w in ref_indices_with_bos_eos]
            # 为BLEU准备的参考词列表 (移除BOS/EOS)
            clean_ref_tokens_for_bleu = []
            if raw_ref_tokens_for_bleu:
                start_idx = 1 if raw_ref_tokens_for_bleu[0] == "BOS" else 0
                end_idx = -1 if raw_ref_tokens_for_bleu[-1] == "EOS" else len(raw_ref_tokens_for_bleu)
                clean_ref_tokens_for_bleu = raw_ref_tokens_for_bleu[start_idx:end_idx]
            ref_text_display = " ".join(clean_ref_tokens_for_bleu)
            print(ref_text_display) # 打印不含BOS/EOS的参考译文
            log_entry_parts.append(ref_text_display)

            # corpus_bleu 需要 list_of_references = [[[ref_tok_sent1]], ...]
            list_of_refs_for_bleu = [[clean_ref_tokens_for_bleu]]

            # --- 模型翻译 ---
            # 模型输入：使用 data.dev_en[i] (包含BOS/EOS的索引)
            src_indices_for_model = data.dev_en[i]
            src = torch.from_numpy(np.array(src_indices_for_model)).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != args.PAD).unsqueeze(-2)

            out = greedy_decode(model, src, src_mask, max_len=args.max_length,
                                start_symbol=data.cn_word_dict["BOS"], device=args.device)

            # 模型翻译结果词元列表 (不含BOS/EOS)，用于显示和BLEU
            model_translation_tokens_for_bleu = []
            for j in range(1, out.size(1)): # 跳过BOS
                sym_idx = out[0, j].item()
                sym = data.cn_index_dict.get(sym_idx, "UNK")
                if sym == 'EOS':
                    break
                model_translation_tokens_for_bleu.append(sym)
            translation_text_display = " ".join(model_translation_tokens_for_bleu)
            print("translation: %s" % translation_text_display)
            log_entry_parts.append("translation: " + translation_text_display)

            # 准备BLEU计算的候选句
            hypotheses_for_bleu = [model_translation_tokens_for_bleu]

            # --- 计算当前句子的准确率 ---
            # 计算当前句子的准确率
            ref_indices = [data.cn_word_dict.get(token, data.cn_word_dict["UNK"]) for token in clean_ref_tokens_for_bleu]
            pred_indices = [data.cn_word_dict.get(token, data.cn_word_dict["UNK"]) for token in model_translation_tokens_for_bleu]

            # 计算准确率（考虑长度差异）
            min_len = min(len(ref_indices), len(pred_indices))
            correct = sum(1 for i in range(min_len) if ref_indices[i] == pred_indices[i])
            total = len(ref_indices)
            accuracy = correct / total if total > 0 else 0
            all_accuracy_scores.append(accuracy)

            # 在输出BLEU分数之前添加准确率输出
            accuracy_line = f"accuracy: {accuracy:.4f}"
            print(accuracy_line)
            log_entry_parts.append(accuracy_line)

            # --- 计算当前句子的BLEU分数 ---
            bleu_1 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_1_gram, smoothing_function=chencherry.method1)
            bleu_2 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_2_gram, smoothing_function=chencherry.method1)
            bleu_3 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_3_gram, smoothing_function=chencherry.method1)
            bleu_4 = corpus_bleu(list_of_refs_for_bleu, hypotheses_for_bleu, weights=weights_4_gram, smoothing_function=chencherry.method1)

            all_bleu_1_scores.append(bleu_1)
            all_bleu_2_scores.append(bleu_2)
            all_bleu_3_scores.append(bleu_3)
            all_bleu_4_scores.append(bleu_4)

            bleu_scores_line = f"bleu : 1_gram = {bleu_1:.4f}  2_gram = {bleu_2:.4f}  3_gram = {bleu_3:.4f}  4_gram = {bleu_4:.4f}"
            print(bleu_scores_line)
            log_entry_parts.append(bleu_scores_line)

            # 将当前句子的所有日志信息（源、参考、翻译、准确率、BLEU）写入文件
            log("\n".join(log_entry_parts) + "\n\n", timestamp_log_str) # 每个样本后加一个空行分隔

        # --- 所有句子评估完毕，计算平均BLEU分数和准确率 ---
        if all_bleu_1_scores: # 确保列表不为空
            avg_bleu_1 = sum(all_bleu_1_scores) / len(all_bleu_1_scores)
            avg_bleu_2 = sum(all_bleu_2_scores) / len(all_bleu_2_scores)
            avg_bleu_3 = sum(all_bleu_3_scores) / len(all_bleu_3_scores)
            avg_bleu_4 = sum(all_bleu_4_scores) / len(all_bleu_4_scores)

            avg_accuracy = sum(all_accuracy_scores) / len(all_accuracy_scores)
            summary_lines = [
                f"\nAverage accuracy: {avg_accuracy:.4f}",
                f"Average bleu : 1_gram = {avg_bleu_1:.4f}  2_gram = {avg_bleu_2:.4f}  3_gram = {avg_bleu_3:.4f}  4_gram = {avg_bleu_4:.4f}\n"
            ]
            for line in summary_lines:
                print(line)
                log(line, timestamp_log_str, is_final_summary=True)
        else:
            no_samples_msg = "No samples were evaluated to calculate average BLEU.\n"
            print(no_samples_msg)
            log(no_samples_msg, timestamp_log_str, is_final_summary=True)

        log(f"Evaluation finished at: {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}\n", timestamp_log_str, is_final_summary=True)