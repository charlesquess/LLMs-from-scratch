# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0（见 LICENSE.txt）。
# 书籍《从零开始构建大语言模型》的源代码
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库：https://github.com/rasbt/LLMs-from-scratch
#
# 基于第7章代码的最小指令微调实现

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time
import urllib

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 从本地文件导入
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 预分词文本
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### 响应:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """自定义数据批处理函数
    
    参数:
        batch: 数据批次
        pad_token_id: 填充token的ID
        ignore_index: 忽略的索引值
        allowed_max_length: 允许的最大长度
        device: 目标设备
        
    返回:
        处理后的输入张量和目标张量
    """
    # 找出批次中最长的序列
    batch_max_length = max(len(item)+1 for item in batch)

    # 填充并准备输入和目标
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 添加一个<|endoftext|> token
        new_item += [pad_token_id]
        # 将序列填充到max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 截取最后一个token作为输入
        targets = torch.tensor(padded[1:])  # 向右移动1位作为目标

        # 新功能：将目标中除第一个填充token外的其他填充token替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 新功能：可选地截断到最大序列长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """下载并加载文件
    
    参数:
        file_path: 本地文件路径
        url: 远程文件URL
        
    返回:
        加载的数据
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def format_input(entry):
    """格式化输入文本
    
    参数:
        entry: 包含指令和输入的数据条目
        
    返回:
        格式化后的文本
    """
    instruction_text = (
        f"以下是一个描述任务的指令。"
        f"请编写一个适当完成请求的响应。"
        f"\n\n### 指令:\n{entry['instruction']}"
    )

    input_text = f"\n\n### 输入:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制损失曲线
    
    参数:
        epochs_seen: 已训练的epoch数
        tokens_seen: 已处理的token数
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制训练和验证损失随epoch的变化
    ax1.plot(epochs_seen, train_losses, label="训练损失")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="验证损失")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("损失")
    ax1.legend(loc="upper right")

    # 为tokens seen创建第二个x轴
    ax2 = ax1.twiny()  # 创建一个共享y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 用于对齐刻度的不可见图
    ax2.set_xlabel("已处理的token数")

    fig.tight_layout()  # 调整布局
    plot_name = "loss-plot-standalone.pdf"
    print(f"图表保存为 {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(test_mode=False):
    """主函数
    
    参数:
        test_mode: 是否以测试模式运行
    """
    #######################################
    # 打印包版本
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 绘图库
        "tiktoken",    # 分词器
        "torch",       # 深度学习库
        "tqdm",        # 进度条
        "tensorflow",  # 用于OpenAI预训练权重
    ]
    for p in pkgs:
        print(f"{p} 版本: {version(p)}")
    print(50*"-")

    #######################################
    # 下载并准备数据集
    #######################################
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85%用于训练
    test_portion = int(len(data) * 0.1)    # 10%用于测试

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # 测试模式下使用很小的子集
    if args.test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("训练集大小:", len(train_data))
    print("验证集大小:", len(val_data))
    print("测试集大小:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)
    print(50*"-")

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # 加载预训练模型
    #######################################

    # 测试用的小型GPT模型
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 120,   # 上下文长度
            "drop_rate": 0.0,        # 丢弃率
            "qkv_bias": False        # 查询-键-值偏置
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "小型测试模型"

    # 主章节中使用的代码
    else:
        BASE_CONFIG = {
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 1024,  # 上下文长度
            "drop_rate": 0.0,        # 丢弃率
            "qkv_bias": True         # 查询-键-值偏置
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "gpt2-medium (355M)"

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(device)

    print("加载模型:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # 微调模型
    #######################################
    print("初始损失")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   训练损失:", train_loss)
    print("   验证损失:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成，耗时 {execution_time_minutes:.2f} 分钟。")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # 保存结果
    #######################################
    print("生成响应")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-response-standalone.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent"用于美化输出
    print(f"响应保存为 {test_data_path}")

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"模型保存为 {file_name}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="微调GPT模型用于分类任务"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("此标志以测试模式运行模型用于内部测试。"
              "否则，将按照章节中的方式运行模型（推荐）。")
    )
    args = parser.parse_args()

    main(args.test_mode)
