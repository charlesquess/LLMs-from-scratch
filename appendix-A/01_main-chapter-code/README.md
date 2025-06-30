# 附录A: PyTorch简介

### 主章节代码

- [code-part1.ipynb](code-part1.ipynb) 包含章节中A.1到A.8部分的所有代码
- [code-part2.ipynb](code-part2.ipynb) 包含章节中A.9 GPU部分的所有代码
- [DDP-script.py](DDP-script.py) 包含演示多GPU使用的脚本(注意Jupyter Notebook仅支持单GPU，所以这是一个脚本而非notebook)。您可以运行`python DDP-script.py`。如果您的机器有超过2个GPU，可以运行`CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py`。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习解答

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是`DDP-script.py`的可选版本，使用PyTorch的`torchrun`命令而非通过`multiprocessing.spawn`自行管理多进程。`torchrun`命令的优势在于自动处理分布式初始化，包括多节点协调，略微简化了设置过程。您可以通过`torchrun --nproc_per_node=2 DDP-script-torchrun.py`运行此脚本
