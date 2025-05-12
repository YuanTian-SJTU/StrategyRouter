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

* `bin_packing` - 包含装箱任务基本实现程序`spec.py`。在应用于其他问题时，**应替换为对应问题的基本实现程序**。（基本实现程序是指，针对该问题的**可执行的、模块化的**程序内容，已经具有**独立的待优化函数**和可**执行的评估函数**）
* `implementation` - 包含 Funsearch 流程中各环节的代码，如`sampler`、`evaluator`、`program_database`等。
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
* `strategy_tracker` - 用于分类和引导 LLM 基于多样的策略生成函数

## 在本地运行 FunSearch 演示

### 参数和设置

如果要调整以下参数，应手动修改 `funsearch/implementation` 中的代码。

* `_reduce_score` - 此函数对某些实例中采样函数的分数进行降维。默认实现为 _mean_。您可以在 `implementation/program_database.py` 中修改它，在那里您可以找到 '\_reduce\_score' 函数。
* `_functions_per_prompt` - 在每一轮的提示词中展示的过往函数的数量，默认值为2，在`implementation/config.py` 中修改。
* `num_islands` - 聚类的岛屿数量，默认值为10，在 `implementation/config.py` 中修改。
* `num_samplers` - 采样器的数量，默认值为1，在 `implementation/config.py` 中修改，当程序并行执行时例如在分布式系统中运行时，您可以将其设置为大于1的值。
* `num_evaluators` - 评估器的数量，默认值为1，在 `implementation/config.py` 中修改，当程序并行执行时例如在分布式系统中运行时，您可以将其设置为大于1的值。
* `reset_period` - 在程序池中，表现最差的聚类被重置的周期（单位为秒），在 `implementation/config.py` 中修改。
* `samples_per_prompt` - 每个提示词生成的独立的样本数量，默认值为4，在 `implementation/config.py` 中修改。

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

1. 根据您的 API 提供商设置 API 的 IP 地址和正确的模型。代码在 `funsearch_bin_packing_llm_api.py` 第 98 - 108 行。

```python
conn = http.client.HTTPSConnection("YOUR API SE RVER IP")
payload = json.dumps({
    "max_tokens": 512,
    "model": "YOUR MODEL NAME",
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ]
})
```

2. 在请求头中设置 API 密钥，代码位于 `funsearch_bin_packing_llm_api.py` 第 109 - 113 行。您应该将 `sk-...` 替换为您的 API 密钥。

```python
headers = {
  'Authorization': 'Bearer sk-YOUR_API_KEY',
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