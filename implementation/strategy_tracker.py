class StrategyTracker:
    def __init__(self):
        self._strategy_scores: dict = {
            "hybrid": [],
            "first_fit": [],
            "best_fit": [],
            "worst_fit": [],
            "greedy": [],
            "other": []
        }
        self._strategy_examples: dict = {
            "hybrid": [],
            "first_fit": [],
            "best_fit": [],
            "worst_fit": [],
            "greedy": [],
            "other": []
        }
        self._round = 0
        self._scores_history: list[dict[str, float]] = []
        self._overall_scores: list[float] = []

    def classify_strategy(self, code: str) -> str:
        """Classify the strategy used in the code."""
        code = code.lower()
        if "hybrid" in code:
            return "hybrid"
        elif "first" in code:
            return "first_fit"
        elif "worst" in code:
            return "worst_fit"
        elif "greedy" in code:
            return "greedy"
        elif "best" in code:
            return "best_fit"
        else:
            return "other"
    
    def update_score(self, code: str, score: float):
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

        return self._strategy_scores
    
    def _get_strategy_prompt(self) -> str:
        """Generate a prompt section about strategy performance."""
        if not self._strategy_scores:
            return ""
        
        prompt = "\nStrategy Performance History:\n"
        for strategy, scores in self._strategy_scores.items():
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            prompt += f"- {strategy}: Average Score = {avg_score:.2f}, Best Score = {max_score:.2f}\n"
        return prompt
