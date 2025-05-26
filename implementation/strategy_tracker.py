from tsp.config_tsp import STRATEGY_KEYWORDS, STRATEGIES

class StrategyTracker:
    def __init__(self):
        self._strategy_scores: dict = {
            "Hybrid": [],
            "Other": []
        }
        self._strategy_examples: dict = {
            "Hybrid": [],
            "Other": []
        }
        for strategy in STRATEGIES:
            self._strategy_scores[strategy] = []
            self._strategy_examples[strategy] = []
        self._round = 0
        self._scores_history: list[dict[str, float]] = []
        self._overall_scores: list[float] = []

    def classify_strategy(self, code: str) -> str:
        """Classify the strategy used in the code."""
        code = code.lower()
        if "hybrid" in code or "combin" in code:    # "combin" for combine, combined and combination
            return "Hybrid"
        for standard_name, keywords in STRATEGY_KEYWORDS.items():
            for kw in keywords:
                if kw in code:
                    return standard_name
        return "Other"
    
    def update_score(self, code: str, score: float) -> [dict, str]:
        """Update the score for a strategy."""
        strategy = self.classify_strategy(code)
        print(f"Classified strategy: {strategy}")
        self._strategy_scores[strategy].append(score)
        self._strategy_examples[strategy].append(code)
        
        # Print current round's scores
        print(f"Sample {self._round} Strategy Scores:")
        print("-" * 50)
        for s, scores in self._strategy_scores.items():
            if scores:
                print(f"{s}: Current Score = {scores[-1]:.2f}, Best Score = {max(scores):.2f}")
        print("-" * 50)
        print('\n')
        self._round += 1

        return self._strategy_scores, strategy
