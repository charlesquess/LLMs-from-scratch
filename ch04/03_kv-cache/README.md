# Bonus Material: KV Cache



# 附加材料: KV缓存

**本文件夹实现了为GPT模型添加KV缓存的功能**

&nbsp;
## 概述

简而言之，KV缓存存储了中间键(K)和值(V)的计算结果以便在推理过程中重复使用，这可以显著提高生成响应时的速度。缺点是它增加了代码复杂度，提高了内存使用量，并且不能在训练时使用。然而，在部署LLM时，推理速度的提升通常值得这些代码复杂度和内存的权衡。

&nbsp;
## 工作原理

想象LLM正在生成一些文本。具体来说，假设LLM收到以下提示："Time flies"。

下图展示了使用第3章修改后的图形来显示注意力分数计算过程，其中键和值向量被高亮显示：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

正如我们在第2章和第4章学到的，LLM一次生成一个词(或token)。假设LLM生成了单词"fast"，那么下一轮的提示变为"Time flies fast"。如下图所示：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

通过比较前两图可以看出，前两个token的键和值向量完全相同，在每次生成下一个token的文本时重新计算它们是浪费的。

因此，KV缓存的思想是实现一个缓存机制，存储之前生成的键和值向量以便重用，这有助于我们避免不必要的重复计算。
## KV缓存实现

实现KV缓存有多种方式，主要思想是在每个生成步骤中只计算新生成token的键和值张量。

我选择了一种强调代码可读性的简单实现。我认为最简单的方式就是浏览代码变更来理解实现方式。

本文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py): 取自第3章和第4章的独立代码，实现LLM并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py): 同上，但添加了实现KV缓存所需的修改

你可以：

a. 打开[`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)文件，查看标记新变更的`# NEW`部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 使用你喜欢的文件差异工具比较两个文件的变更：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

以下是实现细节的简要说明。

&nbsp;

### 1. Registering the cache buffers

Inside the `MultiHeadAttention` constructor we add two non-persistent buffers, `cache_k` and `cache_v`, which will hold concatenated keys and values across steps:

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
```

&nbsp;

### 2. Forward pass with `use_cache` flag

Next, we extend the `forward` method of the `MultiHeadAttention` class to accept `use_cache` argument. After projecting the new chunk of tokens into `keys_new`, `values_new` and `queries`, we either initialize the kv cache or append to our cache:

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. Clearing the cache

When generating texts, between independent sequences (for instance to text generation calls) we must reset both buffers, so we also add a cache resetting method the to the `MultiHeadAttention` class:

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. Propagating `use_cache` in the full model

With the changes to the `MultiHeadAttention` class in place, we now modify the  `GPTModel` class. First, we add a position tracking for the token indices to the instructor:

```python
self.current_pos = 0
```

Then, we replace the one-liner block call with an explicit loop, passing `use_cache` through each transformer block:

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

The above change then also requires a small modification to the `TransformerBlock` class to accept the `use_cache` argument:
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

Lastly, we add a model-level reset to `GPTModel` to clear all block caches at once for our convenience:

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. Using the cache in generation

With the changes to the `GPTModel`, `TransformerBlock`, and `MultiHeadAttention`, finally, here's how we use the KV cache in a simple text generation function:

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

Note that we only feed the model the new token in c) via `logits = model(next_idx, use_cache=True)`. Without caching, we feed the model the whole input `logits = model(idx[:, -ctx_len:], use_cache=False)` as it has no stored keys and values to reuse.

&nbsp;

## 简单性能对比

在概念层面介绍了KV缓存后，一个重要的问题是它在一个小例子中的实际表现如何。为了测试这个实现，我们可以运行上述两个代码文件作为Python脚本，这将运行一个124M参数的小型LLM来生成200个新token(以4个token的提示"Hello, I am"开始):

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备M4芯片的Mac Mini(CPU)上，结果如下:

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

可以看到，即使是一个124M参数的小模型和200个token的短序列长度，我们已经获得了约5倍的加速。(请注意这个实现是为了代码可读性优化的，而不是为了CUDA或MPS运行时速度优化的，后者需要预分配张量而不是重新实例化和连接它们。)

**注意:** 模型在两种情况下都生成了"无意义"的文本，如下所示:

> 输出文本: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章将训练模型，你可以在训练好的模型上使用KV缓存(然而KV缓存仅用于推理)来生成连贯的文本。这里我们使用未训练的模型是为了保持代码简单。

更重要的是，`gpt_ch04.py`和`gpt_with_kv_cache.py`实现生成的文本完全相同。这告诉我们KV缓存的实现是正确的——很容易出现索引错误导致结果不一致。


&nbsp;

## KV缓存的优缺点

随着序列长度的增加，KV缓存的优点和缺点在以下方面变得更加明显：

- [优点] **计算效率提高**: 不使用缓存时，第*t*步的注意力需要将新查询与*t*个之前的键进行比较，因此累积工作量呈二次方增长，O(n²)。使用缓存后，每个键和值只需计算一次然后重复使用，将每步总复杂度降低到线性，O(n)。

- [缺点] **内存使用线性增长**: 每个新token都会追加到KV缓存中。对于长序列和较大的LLM，累积的KV缓存会变得更大，可能消耗大量甚至过多的(GPU)内存。作为变通方案，我们可以截断KV缓存，但这会增加更多复杂性(不过在部署LLM时，这通常还是值得的)。



&nbsp;
## KV缓存实现优化

虽然我上面的KV缓存概念实现有助于理解，并且主要针对代码可读性和教育目的，但在实际应用场景中(特别是对于更大的模型和更长的序列长度)需要更仔细的优化。

&nbsp;
### 扩展缓存时的常见问题

- **内存碎片和重复分配**: 如前所示，通过`torch.cat`连续连接张量，会由于频繁的内存分配和重新分配导致性能瓶颈。

- **内存使用线性增长**: 如果没有正确处理，KV缓存大小对于非常长的序列会变得不切实际。

&nbsp;
#### 技巧1: 预分配内存

与其重复连接张量，我们可以根据预期的最大序列长度预分配足够大的张量。这确保了内存使用的一致性并减少了开销。伪代码如下所示:

```python
# 键和值的预分配示例
max_seq_len = 1024  # 预期最大序列长度
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理过程中，我们可以简单地写入这些预分配张量的切片中。

&nbsp;
#### 技巧2: 通过滑动窗口截断缓存

为了避免GPU内存爆炸性增长，我们可以实现一个带有动态截断的滑动窗口方法。通过滑动窗口，我们只保留缓存中最后的`window_size`个token:

```python
# 滑动窗口缓存实现
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实际优化

你可以在[`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py)文件中找到这些优化。

在配备M4芯片的Mac Mini(CPU)上，生成200个token且窗口大小等于上下文长度(保证结果相同)的情况下，代码运行时间对比如下:

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，在CUDA设备上速度优势消失了，因为这是一个很小的模型，设备传输和通信的开销超过了KV缓存对这种小模型的好处。


&nbsp;
## 附加资源

1. [Qwen3 from-scratch KV cache benchmarks](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation) (Qwen3从零开始的KV缓存基准测试)
2. [Llama 3 from-scratch KV cache benchmarks](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation) (Llama 3从零开始的KV缓存基准测试)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 本文档的更详细版本
