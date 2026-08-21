## GRPO在小模型上的实践
从数据集准备到效果测评，本目录以学习和熟悉为主。

### 工具选择
unsloth + vllm +服务器(暂定)

#### 关于unsloth
[unsloth](https://unsloth.ai/)是用于运行和训练大模型的框架，提供Destop，Web Studio以及Core三种下载方式。
如```curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto```
以及
`uv pip install unsloth vllm --torch-backend=auto`。

或者选择容器安装。

官方对微调所需显存，按照模型参数大小，给出了最低限度参考。

官方提供了微调以及强化学习的ipynb运行笔记，并允许在google colab和kaggle平台上利用免费GPU资源进行训练，对于小规模训练还是很友好的。

unsloth提供**动态4-bit量化**：

1. 以`unsloth-bnb-4bit`结尾即表示为Unsloth 动态 4 位量化模型，比标准 BitsAndBytes 4 位模型消耗略多的显存，但提供显著更高的准确性。如没有unsloth，
仅有bnb-4bit，则为标准 BitsAndBytes 4 位模型。也可以通过设置`load_in_4bit=True`开启动态 4 位量化。
2. 通过切块避免离群值影响全局的数值范围（从而影响参数的表达区分度）
3. 对于4-bit的值分配问题，在原有normal分布上根据实际每个权重块的极值做动态计算，进行更精确的分配
>均匀分配INT4：数轴: [-2, -1.75, -1.5, ..., 0, ..., 1.5, 1.75, 2]
>非均匀分配NF4：数轴: [-2, -1.2, -0.7, -0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 0.7, 1.2, 2]
4. 
