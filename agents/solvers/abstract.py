# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.base import BaseAgent, Scratchpad
from typing import Any, Dict, List, Optional, Tuple
import json
import tempfile
import os
import subprocess
import logging
from abc import abstractmethod


# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class AbstractSolver(BaseAgent):
    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Dict = None, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, predecessors, problem_to_solve)
        self.max_attempts = 1
        self.prompt_dict = prompt_dict if prompt_dict is not None else {}
        self.llm = generator

    @abstractmethod
    def __call__(self, data: Any, memory: Scratchpad) -> Dict[str, Any]:
        """Execute the solver on the given data."""
        pass

    @abstractmethod
    def _formalize(self, problem: str) -> str:
        """Convert the problem into formal code."""
        pass

    @abstractmethod
    def _fix_syntax_error(self, code: str, error_msg: str) -> str:
        """Fix syntax errors in the formal code."""
        pass

    @abstractmethod
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from the LLM response."""
        pass

    @abstractmethod
    def _run_solver(self, code: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Run the solver on the formal code."""
        pass