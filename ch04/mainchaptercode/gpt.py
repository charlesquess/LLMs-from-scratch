"""
GPT模型实现代码(第2-4章)
=========================

本文件包含从第2章到第4章讲解的所有相关代码实现，可以作为一个独立脚本运行。
主要包括以下部分：
1. 第2章：数据加载与预处理
2. 第3章：多头注意力机制实现
3. 第4章：Transformer块和GPT模型实现
"""

import tiktoken  # OpenAI的tokenizer库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# 第2章: 数据加载与预处理
#####################################


class GPTDatasetV1(Dataset):
    """GPT数据集类，用于加载和预处理文本数据
    
    参数:
        txt: 输入文本字符串
        tokenizer: 分词器对象
        max_length: 最大序列长度
        stride: 滑动窗口步长
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入token ID序列
        self.target_ids = []  # 存储目标token ID序列

        # 使用tokenizer编码整个文本，允许特殊token <|endoftext|>
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割为重叠的max_length长度序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """获取指定索引的数据样本
        
        返回:
            元组(input_ids, target_ids)
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """创建数据加载器
    
    参数:
        txt: 输入文本
        batch_size: 批次大小
        max_length: 最大序列长度
        stride: 滑动窗口步长(默认max_length//2)
        shuffle: 是否打乱数据
        drop_last: 是否丢弃不完整的批次
        num_workers: 数据加载工作进程数
        
    返回:
        DataLoader对象
    """

    # 初始化tokenizer(使用GPT2的分词器)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 第3章: 多头注意力机制
#####################################
class MultiHeadAttention(nn.Module):
    """多头注意力机制实现
    
    参数:
        d_in: 输入维度
        d_out: 输出维度
        context_length: 上下文长度
        dropout: dropout概率
        num_heads: 注意力头数
        qkv_bias: 是否在QKV投影中添加偏置
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 初始化QKV投影矩阵
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影层
        self.dropout = nn.Dropout(dropout)
        # 注册因果掩码(上三角矩阵)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, num_tokens, d_in)
            
        返回:
            多头注意力输出，形状为(batch_size, num_tokens, d_out)
        """
        b, num_tokens, d_in = x.shape

        # 计算QKV矩阵
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将QKV矩阵拆分为多个头
        # 展开最后一个维度: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力(自注意力)
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 将掩码截断到当前token数并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多头输出，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选投影

        return context_vec


#####################################
# 第4章: Transformer和GPT模型实现
#####################################
class LayerNorm(nn.Module):
    """层归一化实现
    
    参数:
        emb_dim: 嵌入维度
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的平移参数

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量
            
        返回:
            归一化后的张量
        """
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和平移


class GELU(nn.Module):
    """GELU激活函数实现"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量
            
        返回:
            经过GELU激活的张量
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """前馈神经网络实现
    
    参数:
        cfg: 配置字典
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展维度
            GELU(),  # GELU激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 压缩回原维度
        )

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量
            
        返回:
            前馈网络输出
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer块实现
    
    参数:
        cfg: 配置字典
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(  # 多头注意力层
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)  # 前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个层归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 残差连接的dropout

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量
            
        返回:
            Transformer块输出
        """
        # 注意力块残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加原始输入

        # 前馈网络残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加原始输入

        return x


class GPTModel(nn.Module):
    """GPT模型实现
    
    参数:
        cfg: 配置字典
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # token嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层dropout

        # Transformer块堆叠
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出头

    def forward(self, in_idx):
        """前向传播
        
        参数:
            in_idx: 输入token索引，形状为(batch_size, seq_len)
            
        返回:
            预测logits
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # token嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  # 通过所有Transformer块
        x = self.final_norm(x)  # 最终归一化
        logits = self.out_head(x)  # 输出logits
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """简单文本生成函数(贪婪搜索)
    
    参数:
        model: GPT模型实例
        idx: 初始token索引，形状为(batch_size, seq_len)
        max_new_tokens: 要生成的最大token数
        context_size: 模型支持的上下文长度
        
    返回:
        生成的token索引序列
    """
    # idx是当前上下文中的token索引数组，形状为(batch_size, seq_len)
    for _ in range(max_new_tokens):
        # 如果当前上下文超过模型支持的长度，则截取最后context_size个token
        # 例如，如果模型只支持5个token，而上下文长度是10，则只使用最后5个token
        idx_cond = idx[:, -context_size:]

        # 获取模型预测
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步的预测
        # 形状从(batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取logits值最高的token索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 形状(batch, 1)

        # 将新生成的token索引添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # 形状(batch, n_tokens+1)

    return idx


def main():
    """主函数: 初始化模型并生成文本"""
    # GPT-124M模型配置(约1.24亿参数)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小(GPT-2的词汇表大小)
        "context_length": 1024,  # 上下文长度(最大支持的token数)
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # Transformer层数
        "drop_rate": 0.1,        # Dropout概率
        "qkv_bias": False        # 是否在QKV投影中添加偏置
    }

    torch.manual_seed(123)  # 设置随机种子保证可重复性
    model = GPTModel(GPT_CONFIG_124M)  # 初始化模型
    model.eval()  # 设置为评估模式(禁用dropout)

    start_context = "输入中文句子: "  # 初始输入文本

    # 使用GPT-2的分词器编码文本
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度

    # 打印输入信息
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 生成文本
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())  # 解码生成的token

    # 打印输出信息
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
