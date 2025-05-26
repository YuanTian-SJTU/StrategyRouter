import json
import multiprocessing
from typing import Collection, Any
from matplotlib import pyplot as plt
import http.client
import numpy as np
import time

from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
from implementation import strategy_tracker
from tsp.tsp_utils import datasets
from tsp.config_tsp import STRATEGIES

# 读取API密钥
with open('api_key.txt', 'r') as f:
    api_key = f.read()

# 记录分数
overall_best = []   # 全局最佳
local_best = []    # 局部最佳
strategy_list = []
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

def _trim_preface_of_body(sample: str) -> str:
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
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

class LLMAPI(sampler.LLM):
    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)
        self._trim = trim

    def draw_samples(self, prompt: str) -> Collection[str]:
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        global strategy_scores, round_count, current_trigger_probability, trigger_probability_history, fixed_count
        strategy_prompt = ''
        for strategy, scores in strategy_scores.items():
            if scores:
                score = max(scores)
                strategy_prompt += f"{strategy}: Best score {score:.2f}\n"
            else:
                strategy_prompt += f"{strategy}: Unknown\n"
        if round_count > 0:
            if overall_best[-1] != local_best[-1]:
                current_trigger_probability = min(1.0, current_trigger_probability + 0.05)
            else:
                current_trigger_probability = 0.0
        trigger_probability_history.append(current_trigger_probability)
        if pb_list[round_count] < current_trigger_probability:
            total_count = sum(fixed_count.values())
            if total_count > 0:
                weights = [1 / (fixed_count[strategy] + 1) for strategy in selectable_strategies]
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
                'Be creative and you can implement various strategies like Nearest '+selectable_strategies_str+'or other approaches. '
                'You can also combine multiple strategies or create new ones. '
                'Only output the Python code, no descriptions.'
                'In the function docstring, clearly state which strategy you are using.'
                f'Current strategy scores:\n {strategy_prompt}'
            )
        prompt = '\n'.join([content, additional_prompt])
        round_count += 1
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
                    'Authorization': 'Bearer {}'.format(api_key),
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

class Sandbox(evaluator.Sandbox):
    def __init__(self, verbose=False, numba_accelerate=True):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._strategy_tracker = strategy_tracker.StrategyTracker()
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            test_input: str,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            results = None, False
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()
            else:
                results = None, False
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
        global strategy_scores, failed_count
        if results[0] is not None:
            if not overall_best:
                overall_best.append(results[0])
            else:
                overall_best.append(max(overall_best[-1], results[0]))
            local_best.append(results[0])
            strategy_scores, stg = self._strategy_tracker.update_score(program, results[0])
            strategy_list.append(stg)
            failed_count.append(0)
        else:
            overall_best.append(overall_best[-1])
            local_best.append(-float('inf'))
            strategy_scores, stg = self._strategy_tracker.update_score(program, -float('inf'))
            strategy_list.append(stg)
            failed_count.append(1)
        return results
    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate, result_queue):
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
        except  Exception as e:
            print("Sandox error:", e)
            result_queue.put((None, False))

if __name__ == '__main__':
    with open("tsp/spec.py", "r", encoding="utf-8") as f:
        specification = f.read()
    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config_ = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=300)
    tsp_20 = {'tsp_20': datasets['tsp20']}
    global_max_sample_num = 25 * 4
    print("\nStarting FunSearch for TSP with strategy tracking...")
    start_time = time.time()
    funsearch.main(
        specification=specification,
        inputs=tsp_20,
        config=config_,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/funsearch_tsp_llm_api',
    )
    end_time = time.time()
    run_time = end_time - start_time
    print("\nGenerating plots...")
    plt.figure(figsize=(20, 16))
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
    plt.subplot(2, 2, 2)
    plt.plot(range(len(trigger_probability_history)), trigger_probability_history, 'g-', label='Trigger Probability')
    plt.plot(range(len(pb_list[:len(trigger_probability_history)])), pb_list[:len(trigger_probability_history)], 'r--', label='Random Threshold')
    plt.title('Trigger Probability and Random Threshold')
    plt.xlabel('Sample Number')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(False)
    plt.subplot(2, 2, 3)
    failed_list = []
    failed_ratios = []
    for i in range(len(failed_count)):
        failed_list.append(sum(failed_count[:i+1]))
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
    plt.subplot(2, 2, 4)
    if not strategy_scores:
        print("Warning: No strategies were recorded!")
    else:
        strategies = list(strategy_scores.keys())
        max_scores = [max(scores) if scores else -np.inf for scores in strategy_scores.values()]
        samples = [len(scores) for scores in strategy_scores.values()]
        total_score = sum(samples)
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
    plt.savefig('strategy_scores_tsp.png')
    plt.show()
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
    print(f"Total Failed Attempts: {failed_list[-1]} ({failed_ratios[-1]:.2f}%)")
    print(f"Total Time: {run_time:.2f} seconds ({run_time / len(overall_best):.2f} seconds per attempt on average)")
    with open('StrategyRouterTSPData.csv', 'w') as f:
        f.write('Sample Number, Overall Best, Local, Strategy\n')
        for i, score in enumerate(overall_best):
            f.write(f'{i}, {score}, {local_best[i]}, {strategy_list[i]}\n') 