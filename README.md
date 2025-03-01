# FunSearch 实现

这个仓库是对以下论文的实现：

> Romera-Paredes, B. et al. Mathematical discoveries from program search with large language models. _Nature_ (2023)

本项目是基于 [Google DeepMind 的 FunSearch](https://github.com/google-deepmind/funsearch/blob/main/bin_packing/bin_packing.ipynb) 进行修改和完善的版本，原始代码不完整，缺少沙盒内部运行和调用大模型的函数。本仓库是由 [@YuanTian-SJTU](https://github.com/YuanTian-SJTU/Deepmind-Nature-Repro-FunSearch.git) 完成的可运行版本。

## 安装和要求

请注意，**Python 版本必须大于或等于 Python 3.9**，否则实现中使用的 `_ast_` 包将无法正常工作。

如果有足够的 GPU 设备，您可以在本地运行 FunSearch 进行在线装箱问题求解。或者，您也可以尝试使用 LLM 接口在线请求响应。

请安装 `requirements.txt` 中列出的包。

## 项目结构

本项目包含以下独立目录：

* `bin_packing` - 包含装箱任务的示例 Jupyter notebook
* `implementation` - 包含进化算法、代码操作例程和 FunSearch 流程的单线程实现
* `llm-server` - 包含 LLM 服务器的实现，通过监控来自 FunSearch 的请求获取提示，并将推理结果响应给 FunSearch 算法

## `funsearch/implementation` 中的文件

`funsearch/implementation` 中包含以下文件：

* `code_manipulation.py` - 提供修改规范中代码的函数
* `config.py` - 包含 funsearch 的配置
* `evaluator.py` - 修剪来自 LLM 的样本结果，并评估采样函数
* `evaluator_accelerate.py` - 使用 'numba' 库加速评估
* `funsearch.py` - 实现 funsearch 流程
* `profile.py` - 记录采样函数的得分
* `programs_database.py` - 进化采样函数
* `sampler.py` - 向 LLM 发送提示并获取结果

## 在本地运行 FunSearch 演示

### 参数和设置

如果要调整以下参数，应手动修改 `funsearch/implementation` 中的代码。

* `_reduce_score` - 此函数对某些实例中采样函数的分数进行降维。默认实现为 _mean_。您可以在 `implementation/program_database.py` 中修改它，在那里您可以找到 '\_reduce\_score' 函数。

### 使用本地 LLM

1. 首先，启动本地 LLM 服务器。

```bash
# 假设我们在 funsearch 目录（本项目的根目录）
cd llm-server
# 启动 LLM 服务器：python llm_server.py --port 8088 --path [模型路径] --d [GPU ID]
python llm_server.py --port 8088 --path /LLms/CodeLlama-34b --d 0 1 2 3 4 5
```

2. 然后，启动 FunSearch。

```bash
# 运行 FunSearch
python funsearch_bin_packing_local_llm.py
```

您可以通过 _Tensorboard_ 查看日志。请检查 `bin_packing_funsearch_my_template.py` 中定义的 _log\_dir_ 变量，并使用以下指令启动 Tensorboard：

```bash
# 假设我们在 funsearch 目录（本项目的根目录）
cd logs
tensorboard --logdir funsearch_local_llm
```

### 使用 LLM 接口

1. 根据您的 API 提供商设置 API 的 IP 地址。代码在 `funsearch_bin_packing_llm_api.py` 第 33 行。

```python
conn = http.client.HTTPSConnection("api.chatanywhere.com.cn")
```

2. 在请求头中设置 API 密钥，代码位于 `funsearch_bin_packing_llm_api.py` 第 44-48 行。您应该将 `sk-ys...` 替换为您的 API 密钥。

```python
headers = {
  'Authorization': 'Bearer sk-ys02zx...(替换为您的 API 密钥)...',
  'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
  'Content-Type': 'application/json'
}
```

3. 启动 FunSearch。

```bash
# 运行 FunSearch
python funsearch_bin_packing_llm_api.py
```

您可以通过 _Tensorboard_ 查看日志。请检查 `bin_packing_funsearch_my_template.py` 中定义的 _log\_dir_ 变量，并使用以下指令启动 Tensorboard：

```bash
# 假设我们在 funsearch 目录（本项目的根目录）
cd logs
tensorboard --logdir funsearch_llm_api
```

## 问题

如果您在使用代码时遇到任何困难，请随时提交 issue！