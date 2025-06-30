# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0 许可 (详见 LICENSE.txt)
# 书籍《从零开始构建大语言模型》的源代码
#   - 书籍链接: https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 导入标准库
import json  # JSON处理
import numpy as np  # 数值计算
import os  # 操作系统接口
import urllib.request  # URL请求

# import requests  # 备用HTTP请求库(当前未使用)
import tensorflow as tf  # 用于加载GPT-2的TensorFlow检查点
import tiktoken  # OpenAI的tokenizer
import torch  # PyTorch深度学习框架
from tqdm import tqdm  # 进度条显示

# 从本地文件导入
from previous_chapters import GPTModel  # 导入之前章节实现的GPT模型类


def text_to_token_ids(text, tokenizer):
    """将文本转换为token ID张量
    
    参数:
        text: 输入文本字符串
        tokenizer: 分词器对象
        
    返回:
        包含token ID的PyTorch张量，添加了batch维度(第0维)
    """
    encoded = tokenizer.encode(text)  # 使用tokenizer编码文本
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将token ID张量转换回文本
    
    参数:
        token_ids: 包含token ID的张量
        tokenizer: 分词器对象
        
    返回:
        解码后的文本字符串
    """
    flat = token_ids.squeeze(0)  # 移除batch维度
    return tokenizer.decode(flat.tolist())  # 将token ID列表解码为文本


def download_and_load_gpt2(model_size, models_dir):
    """下载并加载GPT-2模型参数
    
    参数:
        model_size: 模型大小，如"124M"
        models_dir: 模型文件存储目录
        
    返回:
        settings: 模型配置参数
        params: 模型权重参数
        
    异常:
        ValueError: 如果模型大小不在允许范围内
    """
    # 验证模型大小是否有效
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"模型大小不在允许范围内: {allowed_sizes}")

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)  # 模型存储目录
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 模型文件基础URL
    filenames = [  # 需要下载的文件列表
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录(如果不存在)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 文件URL
        file_path = os.path.join(model_dir, filename)  # 本地文件路径
        download_file(file_url, file_path)  # 下载文件

    # 加载配置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取TensorFlow检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型配置
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从TF检查点加载参数

    return settings, params


"""
def download_file(url, destination):
    \"\"\"使用requests库下载文件(备用实现)
    
    参数:
        url: 文件下载URL
        destination: 本地保存路径
        
    功能:
        1. 发送流式GET请求下载文件
        2. 检查文件是否已存在且大小匹配
        3. 显示下载进度条
        4. 分块下载并保存文件
    \"\"\"
    # 发送流式GET请求
    response = requests.get(url, stream=True)

    # 从响应头获取文件总大小(默认0如果不存在)
    file_size = int(response.headers.get("content-length", 0))

    # 检查文件是否已存在且大小相同
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在且是最新版本: {destination}")
            return

    # 定义文件块大小(1KB)
    block_size = 1024  

    # 初始化进度条(显示文件名和总大小)
    progress_bar_description = url.split("/")[-1]  # 从URL提取文件名
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写模式打开目标文件
        with open(destination, "wb") as file:
            # 分块迭代文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 写入文件块
"""


def download_file(url, destination):
    """使用urllib下载文件(当前实现)
    
    参数:
        url: 文件下载URL
        destination: 本地保存路径
        
    功能:
        1. 发送HTTP请求下载文件
        2. 检查文件是否已存在且大小匹配
        3. 显示下载进度条
        4. 分块读取并保存文件
    """
    # 发送HTTP请求
    with urllib.request.urlopen(url) as response:
        # 从响应头获取文件总大小(默认0如果不存在)
        file_size = int(response.headers.get("Content-Length", 0))

        # 检查文件是否已存在且大小相同
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"文件已存在且是最新版本: {destination}")
                return

        # 定义文件块大小(1KB)
        block_size = 1024  

        # 初始化进度条(显示文件名和总大小)
        progress_bar_description = os.path.basename(url)  # 从URL提取文件名
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # 以二进制写模式打开目标文件
            with open(destination, "wb") as file:
                # 分块读取并写入文件
                while True:
                    chunk = response.read(block_size)
                    if not chunk:  # 读取完成
                        break
                    file.write(chunk)  # 写入文件块
                    progress_bar.update(len(chunk))  # 更新进度条


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """从TensorFlow检查点加载GPT-2参数
    
    参数:
        ckpt_path: TensorFlow检查点路径
        settings: 模型配置参数
        
    返回:
        包含GPT-2模型参数的嵌套字典结构
        
    处理流程:
        1. 初始化空参数字典，为每层创建空块
        2. 遍历检查点中的每个变量
        3. 加载变量并去除单一维度
        4. 解析变量名确定目标字典位置
        5. 递归创建嵌套字典结构
        6. 将变量数组分配到最终键
    """
    # 初始化参数字典，为每层创建空块
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并去除单一维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名提取相关部分(跳过'model/'前缀)
        variable_name_parts = name.split("/")[1:]

        # 确定变量的目标字典位置
        target_dict = params
        if variable_name_parts[0].startswith("h"):  # 层特定参数
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组分配到最终键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign(left, right):
    """参数赋值辅助函数，检查形状并转换为PyTorch参数
    
    参数:
        left: 目标参数(用于形状检查)
        right: 源参数值
        
    返回:
        PyTorch参数对象
        
    异常:
        ValueError: 如果形状不匹配
    """
    if left.shape != right.shape:
        raise ValueError(f"形状不匹配. 左: {left.shape}, 右: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """将参数加载到GPT模型中
    
    参数:
        gpt: GPT模型实例
        params: 包含模型参数的字典
        
    处理流程:
        1. 加载位置嵌入和token嵌入权重
        2. 遍历所有Transformer块:
           - 加载注意力层的query/key/value权重和偏置
           - 加载注意力输出投影层的权重和偏置
           - 加载前馈网络的权重和偏置
           - 加载层归一化的缩放和偏移参数
        3. 加载最终层归一化和输出头的权重
    """
    # 加载位置嵌入和token嵌入权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # 遍历所有Transformer块
    for b in range(len(params["blocks"])):
        # 加载注意力层的query/key/value权重
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 加载注意力层的query/key/value偏置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载注意力输出投影层的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 加载前馈网络的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 加载层归一化的缩放和偏移参数
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 加载最终层归一化和输出头的权重
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """生成文本序列
    
    参数:
        model: GPT模型
        idx: 初始输入token ID张量
        max_new_tokens: 最大生成token数量
        context_size: 上下文窗口大小
        temperature: 温度参数(控制生成多样性，0.0为确定性)
        top_k: top-k采样参数(保留概率最高的k个token)
        eos_id: 结束符token ID(可选，遇到时提前停止生成)
        
    返回:
        生成的token ID序列
        
    功能:
        1. 使用模型自回归地生成文本
        2. 支持top-k采样和温度调节
        3. 可选的提前停止机制
    """
    # 循环生成每个新token
    for _ in range(max_new_tokens):
        # 截取最近的context_size个token作为上下文
        idx_cond = idx[:, -context_size:]
        
        # 获取模型输出(禁用梯度计算)
        with torch.no_grad():
            logits = model(idx_cond)
            
        # 只关注最后一个时间步的输出
        logits = logits[:, -1, :]

        # top-k采样: 只保留概率最高的k个token
        if top_k is not None:
            # 获取top-k logits和对应的值
            top_logits, _ = torch.topk(logits, top_k)
            # 计算最小保留值
            min_val = top_logits[:, -1]
            # 将小于min_val的logits设为负无穷(概率为0)
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 温度调节: 控制生成多样性
        if temperature > 0.0:
            logits = logits / temperature  # 缩放logits
            
            # 计算softmax概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            
            # 从分布中采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 温度=0时: 直接选择概率最高的token(确定性)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 遇到结束符时提前停止生成
        if idx_next == eos_id:
            break

        # 将新token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def main(gpt_config, input_prompt, model_size):
    """主函数: 加载GPT-2模型并生成文本
    
    参数:
        gpt_config: GPT模型配置字典
        input_prompt: 输入提示文本
        model_size: 模型大小标识符(如"124M")
        
    处理流程:
        1. 设置计算设备(优先使用GPU)
        2. 下载并加载GPT-2模型参数
        3. 初始化GPT模型并加载权重
        4. 设置模型为评估模式
        5. 初始化tokenizer
        6. 生成文本并输出结果
    """
    # 设置计算设备(优先使用GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载并加载GPT-2模型参数
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 初始化GPT模型并加载权重
    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)  # 将模型移动到指定设备
    gpt.eval()  # 设置模型为评估模式

    # 初始化tokenizer并设置随机种子
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)  # 固定随机种子保证可重复性

    # 生成文本
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    # 输出生成的文本
    print("输出文本:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    """脚本入口
    
    功能:
        1. 设置随机种子保证可重复性
        2. 定义模型配置和输入提示
        3. 调用主函数执行文本生成
    """
    # 固定随机种子保证可重复性
    torch.manual_seed(42)

    # 模型选择和输入提示
    CHOOSE_MODEL = "gpt2-xl (1558M)"  # 可选的模型大小
    INPUT_PROMPT = "what can you do for me?"  # 输入提示文本

    # 基础模型配置
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout率
        "qkv_bias": True         # 是否使用query/key/value偏置
    }

    # 不同大小模型的配置参数
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 提取模型大小标识符(去掉括号和描述)
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # 合并基础配置和模型特定配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 调用主函数
    main(BASE_CONFIG, INPUT_PROMPT, model_size)
