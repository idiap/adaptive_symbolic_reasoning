# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from abc import abstractmethod
from typing import Any, Dict, Optional, List

from agents.base import BaseAgent, Scratchpad

import os

class CritiqueModel(BaseAgent):
    def __init__(self, name: str, goal: str, generator:Any, prompt_dict: Optional[Dict[str, Any]] = None, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, predecessors=predecessors, problem_to_solve=problem_to_solve)
        self.llm = generator
        self.prompt_dict = prompt_dict

    @abstractmethod
    def critique(self, *args, **kwargs):
        pass

    @abstractmethod
    def shutdown(self, *args, **kwargs):
        pass

class FormalisationModel(BaseAgent):
    def __init__(self, name: str, goal: str, generator: Any,  prompt_dict: Optional[Dict[str, Any]] = None, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, predecessors=predecessors, problem_to_solve=problem_to_solve)
        self.llm = generator
        self.code = ''
        self.prompt_dict = prompt_dict

    @abstractmethod
    def formalise(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_formalised_kb(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.formalise(*args, **kwargs)

class HardCritiqueModel(CritiqueModel):
    def __call__(self, memory: Scratchpad, **model_args):
        iter_num = memory.read("iteration_number") or 0
        first_pred = self.predecessors[0] if self.predecessors else None
        explanation = memory.read(f"explanation_{first_pred}") if first_pred else memory.read("explanation") or None
        hypothesis = memory.read(f"hypothesis_{first_pred}") if first_pred else memory.read("hypothesis") or None
        premise: str| None = memory.read(f"premise_{first_pred}") if first_pred else memory.read("premise") or None
        if premise is None:
            premise = memory.read(f'statements_{first_pred}') if first_pred else memory.read('statements')
            hypothesis = memory.read(f'goal_{first_pred}') if first_pred else memory.read('goal')
            explanation = premise[-1]
            premise = ' '.join(premise[:-1])
        history_critique_output = memory.read("history_critique_output") or []
        critique_output = self.critique(
                    iteration_number=iter_num,
                    explanation=explanation,
                    hypothesis=hypothesis,
                    premise=premise,
                    max_tokens=2048,
                    **model_args
                )
    
        history_critique_output.append(f'{iter_num} iteration: {critique_output}')

        memory.write(f"critique_outputs_{self.name}", [critique_output,])
        memory.write("history_critique_output", history_critique_output)

        return f"Critique output generated"