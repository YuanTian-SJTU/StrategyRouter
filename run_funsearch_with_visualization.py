import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

# 导入FunSearch相关模块
from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
from implementation import strategy_tracker
import bin_packing_utils

# 导入自定义的LLMAPI和Sandbox类
from funsearch_bin_packing_llm_api import LLMAPI, Sandbox

# 导入可视化模块
from visualization import StrategyVisualization, generate_all_visualizations


def run_funsearch_with_visualization(dataset_name='OR3', 
                                    samples_per_prompt=4, 
                                    max_sample_nums=12,
                                    evaluate_timeout_seconds=300,
                                    visualization_dir=None):
    """运行FunSearch算法并生成可视化结果
    
    Args:
        dataset_name: 数据集名称，默认为'OR3'
        samples_per_prompt: 每个提示生成的样本数量
        max_sample_nums: 最大样本数量
        evaluate_timeout_seconds: 评估超时时间（秒）
        visualization_dir: 可视化结果保存目录，默认为'logs/visualization_{timestamp}'
    """
    print(f"\n开始运行FunSearch算法（带策略跟踪）...")
    print(f"数据集: {dataset_name}")
    print(f"每个提示样本数: {samples_per_prompt}")
    print(f"最大样本数: {max_sample_nums}")
    print(f"评估超时时间: {evaluate_timeout_seconds}秒")
    
    # 读取规范文件
    with open("bin_packing/spec.py", "r", encoding="utf-8") as f:
        specification = f.read()
    
    # 配置FunSearch
    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config_obj = config.Config(samples_per_prompt=samples_per_prompt, 
                              evaluate_timeout_seconds=evaluate_timeout_seconds)
    
    # 准备数据集
    dataset = {dataset_name: bin_packing_utils.datasets[dataset_name]}
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'logs/funsearch_{timestamp}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 初始化全局变量用于记录分数
    global scores_list, strategy_scores
    scores_list = []
    strategy_scores = {
        "hybrid": [],
        "first_fit": [],
        "best_fit": [],
        "worst_fit": [],
        "greedy": [],
        "other": []
    }
    
    # 运行FunSearch
    start_time = time.time()
    funsearch.main(
        specification=specification,
        inputs=dataset,
        config=config_obj,
        max_sample_nums=max_sample_nums,
        class_config=class_config,
        log_dir=log_dir,
    )
    end_time = time.time()
    
    # 打印运行时间
    run_time = end_time - start_time
    print(f"\nFunSearch运行完成！")
    print(f"总运行时间: {run_time:.2f}秒 ({run_time/60:.2f}分钟)")
    
    # 从LLMAPI和Sandbox中获取记录的分数
    from funsearch_bin_packing_llm_api import scores_list, strategy_scores
    
    # 打印最终策略统计信息
    print("\n最终策略统计信息:")
    print("=" * 50)
    for strategy in strategy_scores:
        scores = strategy_scores[strategy]
        if scores:
            print(f"{strategy}:")
            print(f"  最高分: {max(scores):.2f}")
            print(f"  平均分: {sum(scores)/len(scores):.2f}")
            print(f"  尝试次数: {len(scores)}")
            print("-" * 50)
    
    # 生成可视化结果
    if visualization_dir is None:
        visualization_dir = f'logs/visualization_{timestamp}'
    
    print(f"\n正在生成可视化结果...")
    generate_all_visualizations(scores_list, strategy_scores, save_dir=visualization_dir)
    print(f"可视化结果已保存至: {visualization_dir}")
    
    return scores_list, strategy_scores, visualization_dir


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='运行FunSearch算法并生成可视化结果')
    parser.add_argument('--dataset', type=str, default='OR3', help='数据集名称')
    parser.add_argument('--samples_per_prompt', type=int, default=4, help='每个提示生成的样本数量')
    parser.add_argument('--max_samples', type=int, default=12, help='最大样本数量')
    parser.add_argument('--timeout', type=int, default=300, help='评估超时时间（秒）')
    parser.add_argument('--vis_dir', type=str, default=None, help='可视化结果保存目录')
    args = parser.parse_args()
    
    # 运行FunSearch并生成可视化
    scores_list, strategy_scores, vis_dir = run_funsearch_with_visualization(
        dataset_name=args.dataset,
        samples_per_prompt=args.samples_per_prompt,
        max_sample_nums=args.max_samples,
        evaluate_timeout_seconds=args.timeout,
        visualization_dir=args.vis_dir
    )
    
    # 显示一些可视化结果
    print("\n是否显示可视化结果？(y/n)")
    show_vis = input().strip().lower()
    if show_vis == 'y':
        # 创建可视化对象并显示结果
        vis = StrategyVisualization(save_dir=vis_dir)
        vis.plot_comprehensive_dashboard(scores_list, strategy_scores)