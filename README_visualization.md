# 策略性能可视化工具

本项目提供了一套全面的可视化工具，用于分析和展示FunSearch算法中不同策略的性能和演化过程。这些工具可以帮助研究人员更好地理解策略的表现、比较不同策略的优劣，以及观察策略随时间的演变趋势。

## 功能特点

- **多种可视化图表**：提供多种类型的图表，包括折线图、箱线图、热力图和面积图等
- **策略性能对比**：直观展示不同策略的性能差异
- **时间序列分析**：展示策略性能随时间的变化趋势
- **综合仪表盘**：集成多种图表，提供全面的策略性能概览
- **高质量输出**：生成适合论文发表的高分辨率图表

## 文件说明

- `visualization.py`：核心可视化模块，包含各种可视化函数
- `visualization_example.py`：示例脚本，展示如何使用可视化模块
- `run_funsearch_with_visualization.py`：集成脚本，将FunSearch算法与可视化功能结合

## 使用方法

### 1. 直接使用可视化模块

如果您已经有了策略性能数据，可以直接使用`visualization.py`模块生成可视化图表：

```python
from visualization import StrategyVisualization

# 创建可视化对象
vis = StrategyVisualization(save_dir='visualization_results')

# 绘制整体得分进展图
vis.plot_overall_score_progression(scores_list)

# 绘制各策略得分进展图
vis.plot_strategy_scores(strategy_scores)

# 绘制策略性能对比图（箱线图）
vis.plot_strategy_comparison(strategy_scores)

# 绘制策略演化热力图
vis.plot_strategy_evolution_heatmap(strategy_scores)

# 绘制策略优势演变图
vis.plot_strategy_dominance(strategy_scores)

# 绘制综合仪表盘
vis.plot_comprehensive_dashboard(scores_list, strategy_scores)
```

### 2. 使用示例脚本

`visualization_example.py`提供了一个交互式示例，展示如何使用可视化模块：

```bash
python visualization_example.py
```

该脚本提供了以下选项：
- 演示各个可视化函数
- 一次性生成所有可视化图表（使用示例数据）
- 使用真实数据生成可视化图表（如果可用）

### 3. 集成FunSearch算法与可视化

`run_funsearch_with_visualization.py`将FunSearch算法与可视化功能集成，运行算法后自动生成可视化图表：

```bash
python run_funsearch_with_visualization.py --dataset OR3 --samples_per_prompt 4 --max_samples 12 --timeout 300
```

参数说明：
- `--dataset`：数据集名称，默认为'OR3'
- `--samples_per_prompt`：每个提示生成的样本数量，默认为4
- `--max_samples`：最大样本数量，默认为12
- `--timeout`：评估超时时间（秒），默认为300
- `--vis_dir`：可视化结果保存目录，默认为自动生成

## 图表类型说明

### 整体得分进展图

展示FunSearch算法整体得分随样本数量增加的变化趋势，标记出最高得分点。

### 各策略得分进展图

展示不同策略的得分随样本数量增加的变化趋势，可以直观比较不同策略的性能。

### 策略性能对比图（箱线图）

使用箱线图展示不同策略的性能分布，包括最高分、最低分、中位数和四分位数等统计信息。

### 策略演化热力图

使用热力图展示不同策略随时间的演化过程，颜色深浅表示得分高低。

### 策略优势演变图

使用面积图展示不同策略在各个时间点的相对优势比例，反映策略优势的动态变化。

### 综合仪表盘

集成多种图表，提供全面的策略性能概览，包括整体得分进展、各策略得分进展、策略性能分布、策略优势演变和策略统计信息表格。

## 数据格式要求

### scores_list

整体得分列表，包含每个样本的得分，例如：

```python
scores_list = [0.75, 0.78, 0.80, 0.82, 0.85]
```

### strategy_scores

策略得分字典，键为策略名称，值为得分列表，例如：

```python
strategy_scores = {
    "hybrid": [0.78, 0.82, 0.85],
    "first_fit": [0.70, 0.72, 0.75],
    "best_fit": [0.65, 0.68, 0.72],
    "worst_fit": [0.60, 0.62, 0.65],
    "greedy": [0.72, 0.74, 0.76],
    "other": [0.68, 0.70]
}
```

## 自定义设置

您可以通过修改`StrategyVisualization`类的参数来自定义图表的外观和行为：

- `save_dir`：图表保存目录
- `strategy_colors`：策略颜色方案
- `window_size`：热力图的滑动窗口大小

## 注意事项

- 确保安装了所需的依赖库：matplotlib, numpy, pandas, seaborn
- 图表默认保存为PNG格式，分辨率为300 DPI，适合论文发表
- 如果遇到中文显示问题，请确保系统安装了中文字体