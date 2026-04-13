import asyncio
import json
import os
import random
import re
import textwrap
import torch
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

from swift.infer_engine import RequestConfig, TransformersEngine
from swift.infer_engine.protocol import ChatCompletionResponse, ChatCompletionResponseChoice, RolloutInferRequest
from swift.rewards import ORM, AsyncORM, orms, rm_plugins
from swift.rewards.rm_plugin import DefaultRMPlugin
# register context manager(used in gym training)
from swift.rollout.gym_env import ContextManager, Env, context_managers, envs
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.template import Template
from swift.utils import get_logger, to_device

class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


from difflib import SequenceMatcher

class ComplexLatexRewardORM(ORM):
    """
    Reward function for LaTeX OCR tasks.
    Compares generated LaTeX strings with reference solutions and rewards
    both correctness (string similarity) and complexity handling.
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for gen, sol in zip(completions, solution):
            reward = 0.0

            try:
                # Step 1: Normalize LaTeX strings
                gen_clean = re.sub(r"\s+", "", gen)
                sol_clean = re.sub(r"\s+", "", sol)

                # Step 2: Base similarity
                sim = SequenceMatcher(None, gen_clean, sol_clean).ratio()

                # Step 3: Complexity bonus
                # More symbols and functions indicate higher complexity
                complexity_score = self._complexity_bonus(gen_clean)

                # Step 4: Combine
                # reward = min(1.0, sim + 0.2 * complexity_score)  # cap at 1.0
                coef = 0.9
                reward = 1 if sim == 1.0 else coef * sim + (1-coef) * complexity_score

            except Exception:
                reward = 0.0  # If anything goes wrong, assign 0

            rewards.append(reward)

        return rewards

    def _complexity_bonus(self, latex_str: str) -> float:
        """
        Simple complexity heuristic:
        +1 for each of the following (up to a cap):
        - subscript "_"
        - superscript "^"
        - functions like \sin, \cos, \tan, \log
        - fractions \frac
        """
        symbols = ["_", "^", "\\sin", "\\cos", "\\tan", "\\log", "\\frac"]
        count = sum(latex_str.count(sym) for sym in symbols)
        return min(count / 10.0, 1.0)  # cap bonus at 1.0

# 注册奖励函数
orms['external_r1v_acc'] = ComplexLatexRewardORM