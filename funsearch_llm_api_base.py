import json
import multiprocessing
from typing import Collection, Any
import http.client
import numpy as np
import time
from matplotlib import pyplot as plt

from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
from implementation import strategy_tracker


# 基础函数
def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions/symbols/'def' declaration before the function body."""
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        # find the first 'def' statement in the given code
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


class BaseLLMAPI(sampler.LLM):
    """基础LLM API类，提供通用功能"""

    def __init__(self, samples_per_prompt: int, trim=True, api_key=None):
        super().__init__(samples_per_prompt)
        self._trim = trim
        self._api_key = api_key

    def set_api_key(self, api_key):
        self._api_key = api_key

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _call_api(self, prompt):
        """调用API的通用方法"""
        if not self._api_key:
            raise ValueError("API key not set")

        while True:
            try:
                conn = http.client.HTTPSConnection("api.siliconflow.cn")
                payload = json.dumps({
                    "max_tokens": 512,
                    "model": "THUDM/GLM-4-32B-0414",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': f'Bearer {self._api_key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                if self._trim:
                    response = _trim_preface_of_body(response)
                return response
            except Exception:
                continue

    def _draw_sample(self, content: str) -> str:
        """子类必须实现这个方法"""
        raise NotImplementedError("Subclasses must implement _draw_sample")


class BaseSandbox(evaluator.Sandbox):
    """基础沙盒类，提供通用功能"""

    def __init__(self, verbose=False, numba_accelerate=True):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._strategy_tracker = strategy_tracker.StrategyTracker()

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
        """编译并运行函数的通用方法"""
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except Exception as e:
            print("Sandbox error:", e)
            result_queue.put((None, False))

    def _print_verbose_info(self, program, results, **kwargs):
        """打印详细信息的通用方法"""
        if self._verbose:
            print(f'================= Evaluated Program =================')
            program_: code_manipulation.Program = code_manipulation.text_to_program(text=program)
            func_to_evolve_: str = kwargs.get('func_to_evolve', 'priority')
            function_: code_manipulation.Function = program_.get_function(func_to_evolve_)
            function_: str = str(function_).strip('\n')
            print(f'{function_}')
            print(f'-----------------------------------------------------')
            print(f'Score: {str(results)}')
            print(f'=====================================================')
            print(f'\n')


# 通用绘图函数
def generate_plots(overall_best, local_best, trigger_probability_history, pb_list, failed_count, strategy_scores,
                   fixed_count, run_time, filename='strategy_scores.png'):
    """生成通用的结果图表"""
    plt.figure(figsize=(20, 16))

    # First subplot: Overall score (top left)
    plt.subplot(2, 2, 1)
    if overall_best:
        max_score_index = overall_best.index(max(overall_best))
        plt.plot(range(len(overall_best)), overall_best, 'b-', label='Overall Score')
        plt.scatter(max_score_index, overall_best[max_score_index], color='red',
                    label=f'Max Score ({max_score_index}, {overall_best[max_score_index]:.2f})')
    plt.title('Overall Score Progression')
    plt.xlabel('Sample Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(False)

    # Second subplot: Trigger probability and pb_list (top right)
    plt.subplot(2, 2, 2)
    plt.plot(range(len(trigger_probability_history)), trigger_probability_history, 'g-', label='Trigger Probability')
    plt.plot(range(len(pb_list[:len(trigger_probability_history)])), pb_list[:len(trigger_probability_history)], 'r--',
             label='Random Threshold')
    plt.title('Trigger Probability and Random Threshold')
    plt.xlabel('Sample Number')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(False)

    # Third subplot: Failed count statistics (bottom left)
    plt.subplot(2, 2, 3)
    failed_list = []
    failed_ratios = []
    for i in range(len(failed_count)):
        failed_list.append(sum(failed_count[:i + 1]))
        failed_ratios.append(failed_list[-1] / len(failed_list) * 100)

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(range(len(failed_list)), failed_list, 'b-', label='Failed Count')
    ax2.plot(range(len(failed_ratios)), failed_ratios, 'r-', label='Failed Ratio (%)')

    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Failed Count', color='b')
    ax2.set_ylabel('Failed Ratio (%)', color='r')

    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Failed Attempts Statistics')
    plt.grid(False)

    # Fourth subplot: Strategy scores (bottom right)
    plt.subplot(2, 2, 4)

    if not strategy_scores:
        print("Warning: No strategies were recorded!")
    else:
        # Prepare data for the pie chart
        strategies = list(strategy_scores.keys())
        max_scores = [max(scores) if scores else -np.inf for scores in strategy_scores.values()]
        samples = [len(scores) for scores in strategy_scores.values()]

        # 根据samples占比绘制饼图，在图例中标注每种策略的最高分
        wedges, texts, autotexts = plt.pie(
            samples,
            labels=strategies,
            autopct=lambda p: f'{p:.1f}%',
            startangle=140,
        )
        for i, a in enumerate(autotexts):
            a.set_text(f'{strategies[i]}: {max_scores[i]:.2f}, {samples[i]}')
            a.set_color('black')
            a.set_fontsize(10)
        plt.setp(texts, size=10)
        plt.setp(autotexts, size=10)
        plt.axis('equal')
        plt.title('Strategy Score Distribution (Best Score, Sample Count)')
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    # Print final strategy statistics
    print("\nFinal Strategy Statistics:")
    print("=" * 50)
    for strategy in strategy_scores:
        scores = strategy_scores[strategy]

        if scores:
            print(f"{strategy}:")
            print(f"  Best Score: {max(scores):.2f}")
            print(f"  Best Attempt: {scores.index(max(scores))}")
            print(f"  Average Score: {sum(scores) / len(scores):.2f}")
            print(f"  Number of Total Attempts: {len(scores)}")
            if strategy in fixed_count:
                print(f"  Number of Attempts with Fixed Strategy: {fixed_count[strategy]}")
            print("-" * 50)

    print("=" * 50)
    print("Summary:\n")
    print(f"Total Samples: {len(overall_best)}")
    print(f"Best Overall Score: {max(overall_best):.2f}")
    print(f"Best Overall Attempt: {overall_best.index(max(overall_best))}")
    failed_list = [sum(failed_count[:i + 1]) for i in range(len(failed_count))]
    failed_ratios = [failed_list[-1] / len(failed_count) * 100]
    print(f"Total Failed Attempts: {failed_list[-1]} ({failed_ratios[-1]:.2f}%)")
    print(f"Total Time: {run_time:.2f} seconds ({run_time / len(overall_best):.2f} seconds per attempt on average)")