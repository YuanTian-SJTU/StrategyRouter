# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time
import random

from implementation import evaluator
from implementation import programs_database


class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    RZ: The sampled function code must be trimmed! Especially using instruct-based LLM.
    -For example, the sampled function code (with description) is:
    ------------------------------------------------------------------------------------------------------------------
    Here is the function.
    def priority_v2(..., ...) -> Any:
        a = np.array([1, 2, 3])
        if len(a) > 2:
            return a / a.sum()
        else:
            return a / a.mean()
    This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    -The descriptions above the function's signature, and the function's signature must be removed.
    -The above code must be trimmed as follows:
    ------------------------------------------------------------------------------------------------------------------
        a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        Here is the function. This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    Please note that the indent must be preserved. And the additional descriptions can also be preserved,
    which will be trimmed by Evaluator.
    """

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Node that samples program continuations and sends them for analysis.
    """
    _global_samples_nums: int = 1  # RZ: this variable records the global sample nums
    
    # Predefined priority functions
    _predefined_functions = {
        'first_fit': '''
    """Returns priority with which we want to add item to each bin with First Fit strategy.

    The First Fit strategy places items in the first bin that can accommodate them.
    This strategy is simple and fast but may not always lead to optimal bin utilization.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    valid_bins = bins >= item
    priorities = np.where(valid_bins, 1, 0)
    return priorities
''',
        'best_fit': '''
    """Returns priority with which we want to add item to each bin with Best Fit strategy.

    The Best Fit strategy places items in the bin that leaves the smallest residual space.
    This strategy aims to minimize space wastage but may lead to early bin exhaustion.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    residual_space = bins - item
    valid_bins = residual_space >= 0
    priorities = np.where(valid_bins, -residual_space, np.inf)
    return priorities
''',
        'worst_fit': '''
    """Returns priority with which we want to add item to each bin with Worst Fit strategy.

    The Worst Fit strategy places items in the bin that leaves the largest residual space.
    This strategy aims to balance bin utilization but may lead to inefficient packing.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    residual_space = bins - item
    valid_bins = residual_space >= 0
    priorities = np.where(valid_bins, residual_space, -np.inf)
    return priorities
''',
        'hybrid_fit': '''
    """Returns priority with which we want to add item to each bin with Hybrid Fit strategy.

    The Hybrid Fit strategy combines elements of Best Fit and Worst Fit to find a balance between 
    minimizing space wastage and maximizing utilization. It prioritizes bins that can accommodate 
    the item with the smallest residual space (Best Fit) but also considers bins that have the largest 
    residual space after addition (Worst Fit) to avoid early bin exhaustion.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    residual_space = bins - item
    valid_bins = residual_space >= 0
    
    best_fit_scores = np.where(valid_bins, -residual_space, np.inf)
    worst_fit_scores = np.where(valid_bins, residual_space, -np.inf)
    
    hybrid_scores = (best_fit_scores + worst_fit_scores) / 2
    return hybrid_scores
''',
        'greedy': '''
    """Returns priority with which we want to add item to each bin with Greedy strategy.

    The Greedy strategy prioritizes bins based on their current utilization ratio.
    It aims to maximize bin utilization while maintaining flexibility for future items.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    utilization = item / bins
    valid_bins = bins >= item
    priorities = np.where(valid_bins, utilization, -np.inf)
    return priorities
'''
    }

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        random.seed(42)  # Set random seed for reproducibility

    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis.
        """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()  # RZ: add _global_sample_nums
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1
