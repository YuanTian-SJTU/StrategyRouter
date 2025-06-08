import numpy as np
from fssp.fssp_utils import get_processing_times, calculate_makespan


def fssp_solver(processing_times):
    """Solve the FSSP using priority-based job sequencing."""
    n_jobs, n_machines = processing_times.shape
    unscheduled = list(range(n_jobs))
    job_sequence = []

    while unscheduled:
        priorities = priority(unscheduled, processing_times, job_sequence)
        # 选择优先级最高的作业
        best_job_idx = np.argmax(priorities)
        best_job = unscheduled[best_job_idx]

        job_sequence.append(best_job)
        unscheduled.remove(best_job)

    return job_sequence


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate the average gap between FunSearch and benchmark solutions."""
    total_gap = 0
    solved_instances = 0

    for name, instance in instances.items():
        processing_times = get_processing_times(instance)
        benchmark_makespan = instance['benchmark']

        # Get FunSearch solution
        funsearch_sequence = fssp_solver(processing_times)
        funsearch_makespan = calculate_makespan(funsearch_sequence, processing_times)

        if benchmark_makespan:
            gap = ((funsearch_makespan - benchmark_makespan) / benchmark_makespan * 100)
            total_gap += gap
            solved_instances += 1

    # Return negative average gap (higher is better)
    avg_gap = -total_gap / solved_instances if solved_instances > 0 else -1000
    return avg_gap


@funsearch.evolve
def priority(unscheduled: list, processing_times: np.ndarray, current_sequence: list) -> np.ndarray:
    """
    Assign priorities to unscheduled jobs for Flow Shop Scheduling.

    Args:
        unscheduled: List of unscheduled job indices
        processing_times: Matrix of processing times [job][machine]
        current_sequence: Current partial job sequence

    Returns:
        Array of priorities for unscheduled jobs (higher priority = better)
    """
    n_jobs, n_machines = processing_times.shape
    priorities = np.zeros(len(unscheduled))

    for i, job in enumerate(unscheduled):
        # 基础策略：SPT (Shortest Processing Time)
        total_processing_time = np.sum(processing_times[job])
        priorities[i] = -total_processing_time  # 负数表示时间越短优先级越高

        # 考虑瓶颈机器的影响
        bottleneck_penalty = np.max(processing_times[job]) * 0.5
        priorities[i] -= bottleneck_penalty

    return priorities
