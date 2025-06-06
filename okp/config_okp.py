# 策略名称列表
STRATEGIES = [
    'Greedy',              # 贪心策略
    'Threshold',           # 阈值策略
    'Reservation',         # 预留策略
    'Random',              # 随机策略
    'Adaptive'            # 自适应策略
]

STRATEGY_DESCRIPTIONS = {
    'Greedy': '选择当前价值密度最高的物品',
    'Threshold': '设置价值密度阈值，只接受高于阈值的物品',
    'Reservation': '预留部分容量给未来可能出现的更有价值的物品',
    'Random': '随机决定是否接受当前物品',
    'Adaptive': '根据历史数据动态调整决策策略'
}

STRATEGY_KEYWORDS = {
    'Greedy': ['greedy'],
    'Threshold': ['threshold', 'cutoff'],
    'Reservation': ['reserve', 'future'],
    'Random': ['random'],
    'Adaptive': ['adaptive', 'dynamic']
} 