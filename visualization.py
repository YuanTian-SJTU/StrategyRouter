import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


class StrategyVisualization:
    """用于可视化策略性能和演化过程的类"""

    def __init__(self, save_dir: str = 'visualization_results'):
        """初始化可视化类

        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        
        # 设置默认颜色方案
        self.strategy_colors = {
            "hybrid": "#FF5733",      # 橙红色
            "first_fit": "#33A8FF",  # 蓝色
            "best_fit": "#33FF57",   # 绿色
            "worst_fit": "#FF33A8",  # 粉色
            "greedy": "#A833FF",     # 紫色
            "other": "#FFBD33"       # 黄色
        }
        
        # 设置图表风格
        sns.set_style("whitegrid")
        
    def plot_overall_score_progression(self, scores_list: List[float], 
                                      save_path: Optional[str] = None,
                                      show_plot: bool = True) -> None:
        """绘制整体得分进展图

        Args:
            scores_list: 得分列表
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制得分曲线
        plt.plot(range(len(scores_list)), scores_list, 'b-', linewidth=2, label='整体得分')
        
        # 标记最高分
        if scores_list:
            max_score_index = scores_list.index(max(scores_list))
            plt.scatter(max_score_index, scores_list[max_score_index], color='red', s=100, 
                       label=f'最高分 ({max_score_index}, {scores_list[max_score_index]:.2f})')
        
        # 设置图表属性
        plt.title('策略优化过程中的整体得分进展', fontsize=16)
        plt.xlabel('样本编号', fontsize=14)
        plt.ylabel('得分', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'overall_score_progression.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_strategy_scores(self, strategy_scores: Dict[str, List[float]], 
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
        """绘制各策略得分进展图

        Args:
            strategy_scores: 策略得分字典，键为策略名称，值为得分列表
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
        """
        plt.figure(figsize=(12, 7))
        
        # 绘制每个策略的得分曲线
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                plt.plot(range(len(scores)), scores, '-', linewidth=2, 
                         color=self.strategy_colors.get(strategy, '#999999'),
                         label=f'{strategy}')
                
                # 标记每个策略的最高分
                max_score_index = scores.index(max(scores))
                plt.scatter(max_score_index, scores[max_score_index], 
                           color=self.strategy_colors.get(strategy, '#999999'), s=80)
        
        # 设置图表属性
        plt.title('各策略得分进展对比', fontsize=16)
        plt.xlabel('样本编号', fontsize=14)
        plt.ylabel('得分', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'strategy_scores_progression.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_strategy_comparison(self, strategy_scores: Dict[str, List[float]], 
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
        """绘制策略性能对比图（箱线图）

        Args:
            strategy_scores: 策略得分字典，键为策略名称，值为得分列表
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
        """
        # 准备数据
        data = []
        labels = []
        
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                data.append(scores)
                labels.append(strategy)
        
        if not data:  # 如果没有数据，直接返回
            print("没有可用的策略得分数据")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 创建箱线图
        box_colors = [self.strategy_colors.get(label, '#999999') for label in labels]
        box = plt.boxplot(data, patch_artist=True, labels=labels, showmeans=True)
        
        # 设置箱体颜色
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加数据点（抖动显示）
        for i, d in enumerate(data):
            # 计算抖动
            y = d
            x = np.random.normal(i+1, 0.04, size=len(y))
            plt.scatter(x, y, alpha=0.5, s=30, color=box_colors[i])
        
        # 设置图表属性
        plt.title('各策略性能分布对比', fontsize=16)
        plt.ylabel('得分', fontsize=14)
        plt.xlabel('策略类型', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'strategy_comparison_boxplot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_strategy_evolution_heatmap(self, strategy_scores: Dict[str, List[float]], 
                                      window_size: int = 5,
                                      save_path: Optional[str] = None,
                                      show_plot: bool = True) -> None:
        """绘制策略演化热力图

        Args:
            strategy_scores: 策略得分字典，键为策略名称，值为得分列表
            window_size: 滑动窗口大小，用于计算平均得分
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
        """
        # Prepare data
        strategies = []
        all_scores = []
        
        for strategy, scores in strategy_scores.items():
            if scores:  # Ensure there are scores
                strategies.append(strategy)
                # Use sliding window to calculate average scores
                smoothed_scores = []
                for i in range(len(scores)):
                    start = max(0, i - window_size + 1)
                    smoothed_scores.append(np.mean(scores[start:i+1]))
                all_scores.append(smoothed_scores)
        
        if not all_scores:  # If no data, return directly
            print("No available strategy score data")
            return
        
        # 找到最长的得分列表长度
        max_len = max(len(scores) for scores in all_scores)
        
        # 创建热力图数据矩阵
        heatmap_data = np.zeros((len(strategies), max_len))
        for i, scores in enumerate(all_scores):
            for j, score in enumerate(scores):
                heatmap_data[i, j] = score
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('score_cmap', ['#FFFFFF', '#FFF7BC', '#FEC44F', '#EC7014', '#CC4C02'])
        
        plt.figure(figsize=(14, 8))
        
        # 创建热力图
        sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".1f", 
                   xticklabels=range(max_len), yticklabels=strategies)
        
        # 设置图表属性
        plt.title('策略演化热力图（滑动窗口平均）', fontsize=16)
        plt.xlabel('样本编号', fontsize=14)
        plt.ylabel('策略类型', fontsize=14)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'strategy_evolution_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_strategy_dominance(self, strategy_scores: Dict[str, List[float]], 
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
        """绘制策略优势演变图（面积图）

        Args:
            strategy_scores: 策略得分字典，键为策略名称，值为得分列表
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
        """
        # 准备数据
        strategies = []
        all_scores = []
        max_len = 0
        
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                strategies.append(strategy)
                all_scores.append(scores)
                max_len = max(max_len, len(scores))
        
        if not all_scores:  # 如果没有数据，直接返回
            print("没有可用的策略得分数据")
            return
        
        # 标准化数据长度
        normalized_scores = []
        for scores in all_scores:
            if len(scores) < max_len:
                # 填充缺失的数据点
                scores = scores + [scores[-1]] * (max_len - len(scores))
            normalized_scores.append(scores)
        
        # 创建数据框
        df = pd.DataFrame(normalized_scores).T
        df.columns = strategies
        
        # 计算每个时间点的相对优势（归一化）
        for i in range(len(df)):
            row_sum = df.iloc[i].sum()
            if row_sum > 0:  # 避免除以零
                df.iloc[i] = df.iloc[i] / row_sum
        
        plt.figure(figsize=(12, 7))
        
        # 绘制面积图
        ax = df.plot.area(figsize=(12, 7), alpha=0.7, linewidth=0, 
                         color=[self.strategy_colors.get(s, '#999999') for s in strategies])
        
        # 设置图表属性
        plt.title('策略优势演变图', fontsize=16)
        plt.xlabel('样本编号', fontsize=14)
        plt.ylabel('相对优势比例', fontsize=14)
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'strategy_dominance_area.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_comprehensive_dashboard(self, scores_list: List[float], 
                                   strategy_scores: Dict[str, List[float]],
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> None:
        """绘制综合仪表盘，包含多个图表

        Args:
            scores_list: 整体得分列表
            strategy_scores: 策略得分字典，键为策略名称，值为得分列表
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
        """
        # 创建一个大图和子图布局
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
        
        # 1. 整体得分进展图
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(range(len(scores_list)), scores_list, 'b-', linewidth=2)
        if scores_list:
            max_score_index = scores_list.index(max(scores_list))
            ax1.scatter(max_score_index, scores_list[max_score_index], color='red', s=80)
        ax1.set_title('整体得分进展', fontsize=14)
        ax1.set_xlabel('样本编号', fontsize=12)
        ax1.set_ylabel('得分', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. 各策略得分进展图
        ax2 = plt.subplot(gs[0, 1])
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                ax2.plot(range(len(scores)), scores, '-', linewidth=2, 
                        color=self.strategy_colors.get(strategy, '#999999'),
                        label=f'{strategy}')
        ax2.set_title('各策略得分进展', fontsize=14)
        ax2.set_xlabel('样本编号', fontsize=12)
        ax2.set_ylabel('得分', fontsize=12)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 策略性能对比图（箱线图）
        ax3 = plt.subplot(gs[1, 0])
        data = []
        labels = []
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                data.append(scores)
                labels.append(strategy)
        if data:
            box_colors = [self.strategy_colors.get(label, '#999999') for label in labels]
            box = ax3.boxplot(data, patch_artist=True, labels=labels, showmeans=True)
            for patch, color in zip(box['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax3.set_title('策略性能分布', fontsize=14)
        ax3.set_ylabel('得分', fontsize=12)
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. 策略优势演变图
        ax4 = plt.subplot(gs[1, 1])
        # 准备数据
        strategies = []
        all_scores = []
        max_len = 0
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                strategies.append(strategy)
                all_scores.append(scores)
                max_len = max(max_len, len(scores))
        if all_scores:
            # 标准化数据长度
            normalized_scores = []
            for scores in all_scores:
                if len(scores) < max_len:
                    scores = scores + [scores[-1]] * (max_len - len(scores))
                normalized_scores.append(scores)
            # 创建数据框
            df = pd.DataFrame(normalized_scores).T
            df.columns = strategies
            # 计算每个时间点的相对优势（归一化）
            for i in range(len(df)):
                row_sum = df.iloc[i].sum()
                if row_sum > 0:  # 避免除以零
                    df.iloc[i] = df.iloc[i] / row_sum
            # 绘制面积图
            df.plot.area(ax=ax4, alpha=0.7, linewidth=0, 
                        color=[self.strategy_colors.get(s, '#999999') for s in strategies])
        ax4.set_title('策略优势演变', fontsize=14)
        ax4.set_xlabel('样本编号', fontsize=12)
        ax4.set_ylabel('相对优势比例', fontsize=12)
        ax4.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        ax4.grid(True, alpha=0.3)
        
        # 5. 策略统计信息表格
        ax5 = plt.subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # 准备表格数据
        table_data = []
        table_columns = ['策略', '最高分', '平均分', '最低分', '标准差', '样本数']
        
        for strategy, scores in strategy_scores.items():
            if scores:  # 确保有分数
                row = [
                    strategy,
                    f"{max(scores):.2f}",
                    f"{np.mean(scores):.2f}",
                    f"{min(scores):.2f}",
                    f"{np.std(scores):.2f}",
                    len(scores)
                ]
                table_data.append(row)
        
        if table_data:
            table = ax5.table(cellText=table_data, colLabels=table_columns, 
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            # 设置表头颜色
            for j, key in enumerate(table_columns):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white')
            # 设置策略名称列的颜色
            for i, row in enumerate(table_data):
                strategy = row[0]
                table[(i+1, 0)].set_facecolor(self.strategy_colors.get(strategy, '#999999'))
                table[(i+1, 0)].set_alpha(0.7)
                table[(i+1, 0)].set_text_props(weight='bold')
        
        plt.suptitle('策略性能综合分析仪表盘', fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'comprehensive_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def generate_all_visualizations(scores_list, strategy_scores, save_dir='visualization_results'):
    """Generate all visualization charts
    
    Args:
        scores_list: Overall score list
        strategy_scores: Strategy score dictionary
        save_dir: Save directory
    """
    # Create visualization object
    vis = StrategyVisualization(save_dir=save_dir)
    
    # Generate various charts
    print("Generating overall score progression chart...")
    vis.plot_overall_score_progression(scores_list, show_plot=False)
    
    print("Generating strategy score progression charts...")
    vis.plot_strategy_scores(strategy_scores, show_plot=False)
    
    print("Generating strategy performance comparison chart...")
    vis.plot_strategy_comparison(strategy_scores, show_plot=False)
    
    print("Generating strategy evolution heatmap...")
    vis.plot_strategy_evolution_heatmap(strategy_scores, show_plot=False)
    
    print("Generating strategy dominance evolution chart...")
    vis.plot_strategy_dominance(strategy_scores, show_plot=False)
    
    print("Generating comprehensive dashboard...")
    vis.plot_comprehensive_dashboard(scores_list, strategy_scores, show_plot=False)
    
    print(f"All charts have been saved to {save_dir} directory")


# 示例用法
if __name__ == "__main__":
    # 从funsearch_bin_packing_llm_api.py导入数据
    from funsearch_bin_packing_llm_api import scores_list, strategy_scores
    
    # 生成所有可视化图表
    generate_all_visualizations(scores_list, strategy_scores)