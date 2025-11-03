# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from langchain.schema import HumanMessage, SystemMessage

from base.logger import setup_logging
log = setup_logging()


@dataclass
class Scratchpad:
    """Ephemeral key‑value store with optional TTL."""

    ttl: Optional[timedelta] = None
    _store: Dict[str, Tuple[Any, Optional[datetime]]] = field(default_factory=dict)

    def read(self, key: str) -> Any:
        return self._store.get(key, (None, None))[0]

    def write(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        expiry = datetime.now(timezone.utc) + (ttl or self.ttl) if (ttl or self.ttl) else None
        self._store[key] = (value, expiry)
        log.info("Scratchpad WRITE %s=%s (ttl=%s)", key, str(value)[:60], ttl or self.ttl)

    __getitem__ = read
    __setitem__ = write

Portfolio = Dict[str, "BaseAgent"]


class BaseAgent(ABC):
    """Minimal interface every agent must satisfy."""

    name: str
    goal: str
    predecessors: List[str]
    problem_to_solve: List[str]

    def __init__(self, name: str, goal: str, predecessors: List[str] = None, problem_to_solve: List[str] = None) -> None:
        self.name = name  # This serves as the agent_id
        self.goal = goal
        self.predecessors = predecessors or []
        self.problem_to_solve = problem_to_solve or []
        log.debug("Created agent '%s' with goal: %s, predecessors: %s, problem_to_solve: %s", 
                  name, goal, self.predecessors, self.problem_to_solve)

    def get_problem_key(self) -> str:
        if not self.problem_to_solve:
            return "no_problem"
        return "_".join(sorted(self.problem_to_solve))

    @abstractmethod
    def __call__(self, data: Any, memory: "Scratchpad") -> Any: ...



Input = Any
Output = Any



class LLMToolAgent(BaseAgent):
    """Delegates each call to the chat model via a simple prompt."""

    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Optional[str] = None, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, predecessors, problem_to_solve)
        self.llm = generator
        if prompt_dict is not None:
            self.prompt_dict = prompt_dict
        else:
            self.prompt_dict = {'system_prompt': f"You are agent '{name}'. Goal: {goal}.",
                                'user_prompt': "INPUT:\n[INPUT]\nMEMORY:\n[MEMORY]\n"}

    def __call__(self, memory: Scratchpad, **model_args) -> Output:
        # Read data from the first predecessor if available
        data = memory.read('data')
        if self.predecessors:
            # Use the first predecessor's result as input data
            first_pred = self.predecessors[0]
            pred_result = memory.read(f'result_{first_pred}')
            if pred_result is not None:
                data = pred_result
        
        log.info("[%s] called", self.name)
        user_prompt = self.prompt_dict['user_prompt']
        user_prompt = user_prompt.replace('[INPUT]', str(data))
        user_prompt = user_prompt.replace('[MEMORY]', str(memory._store))
        messages = [
            SystemMessage(content = self.prompt_dict['system_prompt']),
            HumanMessage(content = user_prompt)
        ]
        resp = self.llm(messages, **model_args)
        log.info("[%s] output: %s", self.name, resp)
        
        # Save result using keyword_{agent_name} format
        memory.write(f'result_{self.name}', resp)
        return resp
 

class SpecialControlMarker(LLMToolAgent):
    def __call__(self, memory: Scratchpad, **model_args):
        return 
