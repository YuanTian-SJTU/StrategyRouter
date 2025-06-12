# 策略名称列表
STRATEGIES = [
    'First Fit',           # 首次适应策略
    'Best Fit',            # 最佳适应策略
    'Worst Fit',           # 最差适应策略
    'Next Fit',            # 下一个适应策略
    'Harmonic',            # 谐波策略
]

STRATEGY_DESCRIPTIONS = {
    'First Fit': '将物品放入第一个能容纳它的箱子中',
    'Best Fit': '将物品放入能容纳它且剩余空间最小的箱子中',
    'Worst Fit': '将物品放入能容纳它且剩余空间最大的箱子中',
    'Next Fit': '只考虑当前打开的箱子，如果不能容纳则打开新箱子',
    'Harmonic': '根据物品大小将其分类，并为每类物品分配专用箱子',
}

STRATEGY_KEYWORDS = {
    'First Fit': ['first fit', 'first', 'ff'],
    'Best Fit': ['best fit', 'best', 'bf'],
    'Worst Fit': ['worst fit', 'worst', 'wf'],
    'Next Fit': ['next fit', 'next', 'nf'],
    'Harmonic': ['harmonic', 'harm'],
} 