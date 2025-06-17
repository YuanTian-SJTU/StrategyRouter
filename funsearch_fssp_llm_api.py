from funsearch_llm_api_base import BaseLLMAPI, BaseSandbox, generate_plots
from fssp.fssp_utils import datasets
from fssp.config_fssp import STRATEGIES
import numpy as np

# 读取API密钥
with open('api_key.txt', 'r') as f:
    api_key = f.read()

# 记录分数
overall_best = []
local_best = []
strategy_list = []
# 记录不同策略的分数
strategy_scores = {
    "Hybrid": [],
    "Other": []
}
fixed_count = {}
selectable_strategies = STRATEGIES  # 可选策略
selectable_strategies_str = ''
for strategy in STRATEGIES:
    strategy_scores[strategy] = []
    fixed_count[strategy] = 0
    selectable_strategies_str += (strategy + ', ')
current_trigger_probability = 0.0  # 当前触发概率
trigger_probability_history = []  # 触发历史
failed_count = []
round_count = 0
np.random.seed(10)
pb_list = np.random.random(400)


class LLMAPI(BaseLLMAPI):
    """FSSP特定的LLM API实现"""

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt, trim)
        self.set_api_key(api_key)

    def _draw_sample(self, content: str) -> str:
        # 提取策略分
        global strategy_scores, round_count, current_trigger_probability, trigger_probability_history, fixed_count
        strategy_prompt = ''
        for strategy, scores in strategy_scores.items():
            if scores:
                score = max(scores)
                strategy_prompt += f"{strategy}: Best score {score:.2f}\n"
            else:
                strategy_prompt += f"{strategy}: Unknown\n"

        # 动态调整触发概率
        if round_count > 0:
            if overall_best[-1] != local_best[-1]:
                current_trigger_probability = min(1.0, current_trigger_probability + 0.1)  # 增加触发概率
            else:
                current_trigger_probability = 0.0  # 重置触发概率
        trigger_probability_history.append(current_trigger_probability)

        if pb_list[round_count] < current_trigger_probability:
            # 根据fixed_count动态调整策略选择概率
            total_count = sum(fixed_count.values())
            if total_count > 0:
                # 计算每个策略的权重（使用次数的倒数）
                weights = [1 / (fixed_count[strategy] + 1) for strategy in selectable_strategies]
                # 归一化权重
                weights = np.array(weights) / sum(weights)
                strategy = np.random.choice(selectable_strategies, p=weights)
            else:
                strategy = np.random.choice(selectable_strategies)
            fixed_count[strategy] += 1
        else:
            strategy = None

        if strategy is not None:
            additional_prompt = (
                'Complete a different and more complex Python function. '
                f'You are strongly recommended to use {strategy} strategy. '
                'Only output the Python code, no descriptions.'
                'In the function docstring, clearly state which strategy you are using.'
            )
        else:
            additional_prompt = (
                'Complete a different and more complex Python function. '
                'Be creative and you can implement various strategies like '+selectable_strategies_str+'or other approaches. '
                'You can also combine multiple strategies or create new ones. '
                'Only output the Python code, no descriptions.'
                'In the function docstring, clearly state which strategy you are using.'
                f'Current strategy scores:\n {strategy_prompt}'
            )
        prompt = '\n'.join([content, additional_prompt])
        round_count += 1

        return self._call_api(prompt)


class Sandbox(BaseSandbox):
    """FSSP特定的Sandbox实现"""
    def __init__(self, numba_accelerate=False):
        super().__init__(numba_accelerate=numba_accelerate)

    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: any,
            test_input: str,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded."""
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            # if the process is not finished in time, we consider the program illegal
            process.terminate()
            process.join()
            results = None, False
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()
            else:
                results = None, False

        self._print_verbose_info(program, results, **kwargs)

        # 记录当前最高分和策略分数
        global strategy_scores, failed_count, overall_best, local_best, strategy_list
        if results[0] is not None:  # 如果分数不为空
            # 分数列表
            if not overall_best:  # 如果分数列表为空，直接添加当前分数
                overall_best.append(results[0])
            else:  # 如果分数列表不为空，添加当前分数和列表中上一分数更高的一个
                overall_best.append(max(overall_best[-1], results[0]))
            local_best.append(results[0])
            # 策略分数
            strategy_scores, stg = self._strategy_tracker.update_score(program, results[0])
            strategy_list.append(stg)
            # 记录一个正确函数
            failed_count.append(0)
        else:
            overall_best.append(overall_best[-1] if overall_best else -float('inf'))
            local_best.append(-float('inf'))
            strategy_scores, stg = self._strategy_tracker.update_score(program, -float('inf'))
            strategy_list.append(stg)
            # 记录一个错误函数
            failed_count.append(1)

        return results


if __name__ == '__main__':
    from implementation import config, funsearch
    import time
    import multiprocessing

    with open("fssp/spec.py", "r", encoding="utf-8") as f:
        specification = f.read()

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)

    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=300)

    fssp20x5 = {'fssp20x5': datasets['fssp20x5']}
    global_max_sample_num = 2 * 4  # n * m, n is total number of rounds and m is the number of samplers

    print("\nStarting FunSearch with strategy tracking...")
    # 开始时间
    start_time = time.time()
    funsearch.main(
        specification=specification,
        inputs=fssp20x5,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/funsearch_fssp_llm_api',
    )
    # 结束时间
    end_time = time.time()
    # 计算运行时间
    run_time = end_time - start_time

    print("\nGenerating plots...")
    generate_plots(
        overall_best=overall_best,
        local_best=local_best,
        trigger_probability_history=trigger_probability_history,
        pb_list=pb_list,
        failed_count=failed_count,
        strategy_scores=strategy_scores,
        fixed_count=fixed_count,
        run_time=run_time,
        filename='strategy_scores_fssp.png'
    )

    # 保存分数到csv
    with open('StrategyRouterFSSPData.csv', 'w') as f:
        f.write('Sample Number, Overall Best, Local, Strategy\n')
        for i, score in enumerate(overall_best):
            f.write(f'{i}, {score}, {local_best[i]}, {strategy_list[i]}\n')
