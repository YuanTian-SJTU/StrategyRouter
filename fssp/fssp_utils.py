import numpy as np


def generate_random_fssp_instance(n_jobs, n_machines, min_time=1, max_time=20, seed=None):
    """Generate a random FSSP instance."""
    if seed is not None:
        np.random.seed(seed)

    processing_times = np.random.randint(min_time, max_time + 1, size=(n_jobs, n_machines))
    return processing_times.tolist()


def calculate_lower_bound(processing_times):
    """Calculate a simple lower bound for the makespan."""
    processing_times = np.array(processing_times)
    n_jobs, n_machines = processing_times.shape

    # Lower bound 1: 机器负载下界
    machine_loads = np.sum(processing_times, axis=0)
    lb1 = np.max(machine_loads)

    # Lower bound 2: 关键路径下界
    max_job_time = np.max(np.sum(processing_times, axis=1))
    lb2 = max_job_time

    return max(lb1, lb2)


def get_processing_times(instance):
    """
    Extract processing times matrix from the instance.
    Returns a numpy array where processing_times[job][machine] is the processing time.
    """
    return np.array(instance['processing_times'], dtype=float)


def calculate_makespan(job_sequence, processing_times):
    """Calculate the makespan (total completion time) for a given job sequence."""
    n_jobs, n_machines = processing_times.shape

    # 初始化完成时间矩阵
    completion_times = np.zeros((n_jobs, n_machines))

    for i, job in enumerate(job_sequence):
        for machine in range(n_machines):
            if i == 0 and machine == 0:
                # 第一个作业在第一台机器上
                completion_times[i][machine] = processing_times[job][machine]
            elif i == 0:
                # 第一个作业在其他机器上
                completion_times[i][machine] = completion_times[i][machine - 1] + processing_times[job][machine]
            elif machine == 0:
                # 其他作业在第一台机器上
                completion_times[i][machine] = completion_times[i - 1][machine] + processing_times[job][machine]
            else:
                # 其他作业在其他机器上
                completion_times[i][machine] = max(
                    completion_times[i][machine - 1],  # 同一作业前一台机器完成时间
                    completion_times[i - 1][machine]  # 同一机器前一作业完成时间
                ) + processing_times[job][machine]

    # 返回最后一个作业在最后一台机器上的完成时间
    return completion_times[-1][-1]


# FSSP测试数据集
datasets = {}

# 小规模测试实例 - 5个作业，3台机器
datasets['fssp5x3'] = {
    'fssp5x3_01': {
        'jobs': 5,
        'machines': 3,
        'processing_times': [
            [3, 2, 4],  # Job 0
            [1, 5, 2],  # Job 1
            [4, 1, 3],  # Job 2
            [2, 3, 1],  # Job 3
            [5, 4, 2]  # Job 4
        ],
        'benchmark': 21
    }
}

# 中等规模测试实例 - 10个作业，5台机器
datasets['fssp10x5'] = {
    'fssp10x5_01': {
        'jobs': 10,
        'machines': 5,
        'processing_times': [
            [2, 4, 6, 3, 5],  # Job 0
            [5, 1, 3, 7, 2],  # Job 1
            [3, 6, 2, 4, 8],  # Job 2
            [7, 3, 5, 1, 4],  # Job 3
            [1, 8, 4, 6, 3],  # Job 4
            [4, 2, 7, 5, 1],  # Job 5
            [6, 5, 1, 8, 7],  # Job 6
            [8, 7, 3, 2, 6],  # Job 7
            [2, 1, 8, 4, 5],  # Job 8
            [5, 4, 2, 3, 1]  # Job 9
        ],
        'benchmark': 73
    }
}

# 较大规模测试实例 - 20个作业，5台机器
datasets['fssp20x5'] = {
    'fssp20x5_01': {
        'jobs': 20,
        'machines': 5,
        'processing_times': [
            [54, 83, 15, 71, 77],  # Job 0
            [79, 3, 11, 99, 56],  # Job 1
            [15, 11, 81, 73, 21],  # Job 2
            [61, 89, 74, 18, 39],  # Job 3
            [97, 32, 13, 53, 41],  # Job 4
            [2, 95, 86, 12, 48],  # Job 5
            [14, 73, 82, 37, 91],  # Job 6
            [62, 27, 94, 58, 76],  # Job 7
            [43, 5, 79, 66, 28],  # Job 8
            [16, 44, 52, 88, 35],  # Job 9
            [24, 69, 17, 47, 93],  # Job 10
            [84, 8, 49, 31, 67],  # Job 11
            [25, 75, 6, 85, 19],  # Job 12
            [9, 51, 38, 23, 72],  # Job 13
            [46, 1, 65, 42, 87],  # Job 14
            [33, 59, 22, 96, 4],  # Job 15
            [68, 36, 92, 7, 54],  # Job 16
            [45, 18, 63, 81, 29],  # Job 17
            [57, 74, 26, 40, 95],  # Job 18
            [10, 64, 50, 78, 34]  # Job 19
        ],
        'benchmark': 1574
    },
    'fssp20x5_02': {
        'jobs': 20,
        'machines': 5,
        'processing_times': [
            [21, 53, 95, 55, 34],  # Job 0
            [78, 21, 53, 52, 21],  # Job 1
            [35, 12, 88, 69, 77],  # Job 2
            [55, 48, 17, 83, 40],  # Job 3
            [12, 54, 92, 79, 66],  # Job 4
            [77, 32, 41, 1, 93],  # Job 5
            [62, 68, 47, 44, 28],  # Job 6
            [83, 26, 36, 11, 75],  # Job 7
            [41, 89, 72, 59, 91],  # Job 8
            [17, 35, 61, 98, 22],  # Job 9
            [71, 99, 25, 37, 84],  # Job 10
            [44, 67, 16, 76, 29],  # Job 11
            [99, 24, 89, 60, 45],  # Job 12
            [33, 71, 14, 95, 58],  # Job 13
            [86, 43, 67, 23, 39],  # Job 14
            [19, 85, 31, 74, 90],  # Job 15
            [56, 14, 82, 49, 27],  # Job 16
            [73, 91, 6, 38, 63],  # Job 17
            [28, 57, 94, 15, 81],  # Job 18
            [64, 39, 58, 87, 46]  # Job 19
        ],
        'benchmark': 1604
    }
}