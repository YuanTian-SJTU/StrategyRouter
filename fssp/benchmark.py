"""
最简单的FSSP Benchmark计算
只用一个函数计算所有datasets的optimal值
"""

import numpy as np
from fssp_utils import datasets, calculate_makespan, get_processing_times

def simple_neh(processing_times):
    """简单的NEH算法"""
    n_jobs, n_machines = processing_times.shape
    jobs = list(range(n_jobs))

    # 按总时间降序排列
    total_times = [np.sum(processing_times[job]) for job in jobs]
    jobs.sort(key=lambda x: total_times[x], reverse=True)

    # 插入算法
    sequence = [jobs[0]]
    for job in jobs[1:]:
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(sequence) + 1):
            temp_seq = sequence[:pos] + [job] + sequence[pos:]
            makespan = calculate_makespan(temp_seq, processing_times)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        sequence.insert(best_pos, job)

    return calculate_makespan(sequence, processing_times)


def calculate_benchmarks():
    """计算所有datasets的benchmark值"""
    print("计算FSSP benchmark值:")
    print("-" * 30)

    for dataset_name, dataset in datasets.items():
        print(f"\n{dataset_name}:")
        for instance_name, instance in dataset.items():
            processing_times = get_processing_times(instance)

            # 如果有optimal值就使用，没有就计算
            optimal = simple_neh(processing_times)
            if instance['optimal'] > optimal:
                status = "(计算)"
            else:
                status = "(已存在)"

            print(f"  {instance_name}: {optimal} {status}")


if __name__ == '__main__':
    calculate_benchmarks()