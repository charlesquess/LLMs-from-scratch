# 可选设置说明

本文档列出了设置您的机器和使用本仓库代码的不同方法。我建议从上到下浏览不同部分，然后决定哪种方法最适合您的需求。

&nbsp;

## 快速开始

如果您的机器上已经安装了Python，最快的方法是执行以下pip安装命令从代码仓库的根目录安装[../requirements.txt](../requirements.txt)文件中的包依赖：

```bash
pip install -r requirements.txt
```

<br>

> **注意：** 如果您在Google Colab上运行任何笔记本并想安装依赖项，只需在笔记本顶部的新单元格中运行以下代码：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

在下面的视频中，我分享了在电脑上设置Python环境的个人方法：

<br>
<br>

[![视频链接](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


&nbsp;
# 本地设置

本节提供了在本地运行本书代码的建议。请注意，本书主要章节的代码设计为在常规笔记本电脑上以合理的时间运行，不需要专门的硬件。我在M3 MacBook Air笔记本电脑上测试了所有主要章节。此外，如果您的笔记本电脑或台式机有NVIDIA GPU，代码将自动利用它。

&nbsp;
## 设置Python

如果您的机器上尚未设置Python，我在以下目录中写了关于个人Python设置偏好的内容：

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下面的*使用DevContainers*部分概述了在机器上安装项目依赖项的替代方法。

&nbsp;

## 使用Docker DevContainers

作为上述*设置Python*部分的替代方案，如果您更喜欢隔离项目依赖项和配置的开发设置，使用Docker是一个高效的解决方案。这种方法消除了手动安装软件包和库的需要，并确保一致的开发环境。您可以找到设置Docker和使用DevContainer的更多说明：

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code编辑器

有许多优秀的代码编辑器选择。我首选的流行开源[Visual Studio Code (VSCode)](https://code.visualstudio.com)编辑器，可以轻松增强许多有用的插件和扩展（更多信息请参见下面的*VSCode扩展*部分）。macOS、Linux和Windows的下载说明可以在[VSCode主网站](https://code.visualstudio.com)找到。

&nbsp;

## VSCode扩展

如果您使用Visual Studio Code (VSCode)作为主要代码编辑器，可以在`.vscode`子文件夹中找到推荐的扩展。这些扩展提供了增强的功能和工具，对本仓库有帮助。

要安装这些，在VSCode中打开这个"setup"文件夹（文件 -> 打开文件夹...），然后点击右下角弹出菜单中的"安装"按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

或者，您可以将`.vscode`扩展文件夹移动到此GitHub仓库的根目录：

```bash
mv setup/.vscode ./
```

然后，每次打开`LLMs-from-scratch`主文件夹时，VSCode会自动检查推荐的扩展是否已安装在您的系统上。

&nbsp;

# 云资源

本节描述了运行本书代码的云替代方案。

虽然代码可以在没有专用GPU的传统笔记本电脑和台式计算机上运行，但带有NVIDIA GPU的云平台可以显著提高代码的运行时间，特别是在第5章到第7章中。

&nbsp;

## 使用Lightning Studio

为了在云中获得流畅的开发体验，我推荐[Lightning AI Studio](https://lightning.ai/)平台，它允许用户在云CPU和GPU上设置持久环境并使用VSCode和Jupyter Lab。

启动新Studio后，您可以打开终端并执行以下设置步骤来克隆仓库并安装依赖项：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

（与Google Colab不同，这些只需执行一次，因为Lightning AI Studio环境是持久的，即使您在CPU和GPU机器之间切换。）

然后，导航到您要运行的Python脚本或Jupyter Notebook。可选地，您还可以轻松连接GPU来加速代码运行时间，例如当您在第5章预训练LLM或在第6章和第7章微调它时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用Google Colab

要在云中使用Google Colab环境，请访问[https://colab.research.google.com/](https://colab.research.google.com/)并从GitHub菜单打开相应的章节笔记本，或将笔记本拖到*上传*字段中，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


还要确保将相关文件（数据集文件和笔记本导入的.py文件）上传到Colab环境，如下所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


您可以通过更改*运行时*选择在GPU上运行代码，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;

# 有问题？

如果您有任何问题，请随时通过此GitHub仓库中的[讨论](https://github.com/rasbt/LLMs-from-scratch/discussions)论坛联系我们。
