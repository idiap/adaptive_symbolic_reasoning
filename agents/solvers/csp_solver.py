# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.base import Scratchpad
from typing import Any, Dict, List, Optional, Tuple
import json
import tempfile
import re
import subprocess
import logging
import os
from .abstract import AbstractSolver
from agents.generation.local import LocalGenerator

# Set up logging
from base.logger import setup_logging
log = setup_logging()

class CSPSolver(AbstractSolver):
    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Dict = None, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, generator, prompt_dict, predecessors, problem_to_solve)
        if isinstance(generator, LocalGenerator):
            self.local_flag = True
        else:
            self.local_flag = False
        self.solver_path = 'minizinc'

    def __call__(self, memory: Scratchpad, **model_args) -> Dict[str, Any]:
        try:
            # Read data with first predecessor's name suffix, fallback to generic
            first_pred = self.predecessors[0]
            data = memory.read(f"csp_input_{first_pred}")
            
            if not isinstance(data, dict) or 'problem' not in data:
                raise ValueError("Invalid input format")
            
            if not data['problem']:
                raise ValueError("Problem description is empty")
            
            # First formalize the problem
            code = self._formalize(data['problem'], **model_args)
            memory.write(f"minizinc_code_{self.name}", code)  # Store code even if solving fails
                   
            # Then try to solve it
            for attempt in range(self.max_attempts):
                log.info("Attempt %d/%d", attempt + 1, self.max_attempts)
                success, error_msg, solutions = self._run_solver(code)
                memory.write(f"solutions_{self.name}", solutions)
                
                if success:
                    log.info("MiniZinc ran successfully. Found %d solutions", len(solutions) if solutions else 0)
                    if solutions:
                        result = {
                            'solutions': solutions,
                            'assignments': solutions[0]['assignments'],
                            'minizinc_code': code
                        }
                        memory.write(f"result_{self.get_problem_key()}", result)
                        return result
                    else:
                        log.warning("No solutions found in successful run")
                else:
                    log.warning("MiniZinc error: %s", error_msg)
                    
                if attempt < self.max_attempts - 1:
                    try:
                        log.info("Attempting to fix error...")
                        code = self._fix_syntax_error(code, error_msg, **model_args)
                        log.info("Code fixed, trying again...")
                    except Exception as e:
                        log.error("Error during fix attempt: %s", str(e))
                        if attempt == self.max_attempts - 1:
                            raise ValueError(f"Failed to solve after {self.max_attempts} attempts: {str(e)}")
                        continue
                    
            raise ValueError(f"Failed to solve after {self.max_attempts} attempts")
            
        except Exception as e:
            log.error("Failed to solve CSP: %s", str(e))
            raise ValueError(f"Failed to solve CSP: {str(e)}")

    def _formalize(self, problem: str, output_anyway = False, **model_args) -> str:
        if 'response_format' in model_args:
            del model_args['response_format']
        try:
            del self.llm.client.model_kwargs['response_format']
        except:
            pass
        response = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['generate minizinc from prompt'],
            model_args = model_args,
            premise=problem,
        )
        
        if not response:
            raise ValueError("LLM returned empty response")
        
        code = self._extract_code_from_response(response)
        
        if not code and not output_anyway:
            raise ValueError("No MiniZinc code found in LLM response")

        return code

    def _fix_syntax_error(self, code: str, error_msg: str, **model_args) -> str:
        if 'response_format' in model_args:
            del model_args['response_format']
        refined = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine minizinc error'],
            model_args = model_args,
            code='"""'+code+'"""',
            error=error_msg,
        )
        
        if not refined:
            raise ValueError("LLM returned empty response for refinement")
        
        return self._extract_code_from_response(refined)

    def _extract_code_from_response(self, response: str) -> str:
        """Extract MiniZinc code from LLM response."""
        if not response:
            return ""
            
        # Try to extract code from triple quotes
        code_blocks = response.split("'''")
        if len(code_blocks) >= 3:
            return code_blocks[1].strip()
            
        # Try if response starts with "minizinc"
        if response.lower().startswith("minizinc"):
            return response[len("minizinc"):].strip()
            
        # Try to find code starting with a comment
        if "% minizinc" in response.lower():
            return response[response.lower().find("% minizinc"):].strip()
            
        # Otherwise return the whole response
        return response.strip()

    def _run_solver(self, code: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mzn', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                cmd = [self.solver_path, '--solver', 'gecode', temp_file]
                log.info("Running command: %s", ' '.join(cmd))
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                log.info("MiniZinc stdout:\n%s", result.stdout)
                if result.stderr:
                    log.info("MiniZinc stderr:\n%s", result.stderr)
                
                solutions = self._parse_solutions(result.stdout)
                log.info("Parsed %d solutions", len(solutions))
                
                if solutions:
                    if not isinstance(solutions, list):
                        solutions = [solutions]
                    return True, "", solutions
                else:
                    return False, "No valid solutions found", []
                    
            except subprocess.CalledProcessError as e:
                log.error("MiniZinc process error: %s", e.stderr)
                return False, e.stderr, []
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            log.error("Error running MiniZinc: %s", str(e))
            return False, str(e), []

    def _parse_solutions(self, output: str) -> List[Dict[str, Any]]:
        solutions = []
        solution_texts = output.split('----------')
        for solution_text in solution_texts:
            if not solution_text.strip() or '==========' in solution_text:
                continue
            # Look for the format: variable_name = [item1: value1, item2: value2, ...];
            match = re.search(r'(\w+)\s*=\s*\[(.*?)\];', solution_text, re.DOTALL)
            if match:
                variable_name = match.group(1)
                assignments_str = match.group(2)
                raw_items = [item.strip() for item in assignments_str.split(',') if item.strip()]
                assignments: Dict[str, Any] = {}
                # Detect whether MiniZinc printed explicit indices ("1: value")
                has_named_items = any(':' in item for item in raw_items)

                if has_named_items:
                    for item in raw_items:
                        if ':' not in item:
                            continue
                        key, value = item.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            assignments[key] = int(value)
                        except ValueError:
                            assignments[key] = value
                else:
                    # MiniZinc often emits vectors like [Poster_1, OM_A, ...].
                    # Preserve ordering so downstream components can align
                    # with domain objects (e.g., papers P1..Pn).
                    assignments = {str(idx + 1): value for idx, value in enumerate(raw_items)}

                solutions.append({'assignments': {variable_name: assignments}})
        return solutions
