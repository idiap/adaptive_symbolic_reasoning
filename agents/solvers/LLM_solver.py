# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.base import Scratchpad, BaseAgent
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
import logging
import time

# Set up logging
from base.logger import setup_logging
log = setup_logging()

label_map = {'correct':True, 'incorrect':False, 'parsing_error':False}

class LLMSolver(BaseAgent):
    def __init__(self, name: str, goal: str, prompt_dict: Dict, generator: Any, expect_json = True, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, predecessors, problem_to_solve)
        self.expect_json = expect_json
        self.prompt_dict = prompt_dict
        self.max_attempts = 2
        self.llm = generator

    def __call__(self, memory: Scratchpad, **model_args) -> Dict[str, Any]:
        try:
            # Read data with first predecessor's name suffix
            first_pred = self.predecessors[0]
            statement = memory.read(f"statement_{first_pred}") or None
            premise: str| None = memory.read(f"premise_{first_pred}") or None
            
            if premise is None or statement is None:
                raise ValueError("Incomplete input")


            if isinstance(statement, list):
                statement = ' '.join(statement)

            if isinstance(premise, list):
                premise = ' '.join(premise)
            

            solution_history = []
            overall_verification = False
            
            # solve the problem
            solution_solver = self.llm.generate(
                    model_prompt_dir="./",
                    prompt_name=self.prompt_dict['get_solution'],
                    model_args={'response_format': {"type": "json_object"}},
                    PREMISE=premise,
                    STATEMENT=statement
                )
            if not isinstance(solution_solver, dict):
                try:
                    solution_solver = json.loads(solution_solver)
                except json.JSONDecodeError as e:
                    log.error("Solution JSON error: %s", solution_solver)
                    raise RuntimeError("Solver emitted invalid JSON") from e
            reasoning = solution_solver['reasoning']
            pred = solution_solver['label']


            # verify reasoning
            solution_verifier = self.llm.generate(
                    model_prompt_dir="./",
                    prompt_name=self.prompt_dict['get_verification'],
                    model_args={'response_format': {"type": "json_object"}},
                    PREMISE=premise,
                    STATEMENT=statement,
                    SOLVER_JSON=str(solution_solver)
                )
            if not isinstance(solution_verifier, dict):
                try:
                    solution_verifier = json.loads(solution_verifier)
                except json.JSONDecodeError as e:
                    log.error("Verification JSON error: %s", solution_verifier)
                    raise RuntimeError("Verifier emitted invalid JSON") from e
            reasoning_verification = label_map.get(solution_verifier.get('pattern_verification', 'reasoning_verification_error'), False)
            fact_verification = label_map.get(solution_verifier.get('fact_verification', 'fact_verification_error'), False)
            overall_verification = fact_verification and reasoning_verification

            solution_final = {}
            solution_final["reasoning"] = reasoning
            solution_final["pred"] = pred
            solution_final["fact_verification"] = solution_verifier.get('fact_verification', 'fact_verification_error')
            solution_final["reasoning_verification"] = solution_verifier.get('pattern_verification', 'reasoning_verification_error')
            solution_final["overall_verification"] = overall_verification
            solution_history.append(solution_final)

            # Refine the solution
            if not overall_verification:
                solution_refiner = self.llm.generate(
                        model_prompt_dir="./",
                        prompt_name=self.prompt_dict['get_refinement'],
                        model_args={'response_format': {"type": "json_object"}},
                        PREMISE=premise,
                        STATEMENT=statement,
                        SOLVER_JSON=str(solution_solver),
                        VERIFIER_JSON=str(solution_verifier)
                    )
                if not isinstance(solution_refiner, dict):
                    try:
                        solution_refiner = json.loads(solution_refiner)
                    except json.JSONDecodeError as e:
                        log.error("Refinement JSON error: %s", solution_refiner)
                        raise RuntimeError("Refiner emitted invalid JSON") from e
                reasoning = solution_refiner['reasoning']
                pred = solution_refiner['label']

                solution_final = {}
                solution_final["reasoning"] = reasoning
                solution_final["pred"] = pred
                solution_history.append(solution_final)

            memory.write(f"solution_history", solution_history)
            memory.write(f"solution_final", solution_final)
            
            return overall_verification
                    
        except Exception as e:
            log.error("Failed to solve reasoning problem: %s", str(e))
            raise ValueError(f"Failed to solve reasoning problem: {str(e)}")
        