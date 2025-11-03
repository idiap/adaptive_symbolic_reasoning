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
        # self.solver_path = "/Applications/MiniZincIDE.app/Contents/Resources/minizinc"
        if isinstance(generator, LocalGenerator):
            self.local_flag = True
        else:
            self.local_flag = False
        self.solver_path = 'minizinc'
        self.ordering_str = ''
    
    def _extract_answer(self, model_output):
        """
        Extracts the answer letter from a line like 'ANSWER: D' or '$\\boxed{D}$' in the model output.
        Returns the letter as a string, or None if not found.
        """
        # Try the expected ANSWER: format first
        match = re.search(r'ANSWER:\s*([A-G])', model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Try LaTeX boxed format that Gemini seems to be using
        match = re.search(r'\\boxed\{([A-G])\}', model_output)
        if match:
            return match.group(1).upper()
        
        # Try just the letter in boxed format without LaTeX
        match = re.search(r'boxed\{([A-G])\}', model_output)
        if match:
            return match.group(1).upper()
        
        return None

    def _get_final_result(self, memory, **model_args):
        csp_order = memory.read(f"csp_order_{self.name}")
        solutions = memory.read(f"solutions_{self.name}")   
        # Read options from first predecessor if available
        first_pred = self.predecessors[0]
        options = memory.read(f"options_{first_pred}") or None
        model_output = self.llm.generate(
            model_prompt_dir="helpers",
            prompt_name="match_csp_solution.txt",
            model_args = model_args,
            solution=solutions[0]['assignments'],
            options=options,
            csp_order=csp_order
        )
        parsed_answer = self._extract_answer(model_output) 

        return parsed_answer

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
            order, code = self._formalize(data['problem'], **model_args)    
            memory.write(f"csp_order_{self.name}", order) 
                   
            # Then try to solve it
            for attempt in range(self.max_attempts):
                log.info("Attempt %d/%d", attempt + 1, self.max_attempts)
                success, error_msg, solutions = self._run_solver(code)
                memory.write(f"solutions_{self.name}", solutions)
                
                if success:
                    log.info("MiniZinc ran successfully. Found %d solutions", len(solutions) if solutions else 0)
                    solver_label = self._get_final_result(memory)
                    if solutions:
                        result = {
                            'ori_answer': solutions,
                            'parsed_answer': solver_label,
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
        
        order, code = self._extract_code_from_response(response)
        
        if not code and not output_anyway:
            raise ValueError("No MiniZinc code found in LLM response")
        

        return order, code

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
        if not response:
            return ""

        pattern = r'<mean>(.*?)</mean>'
        try:
            self.ordering_str = re.findall(pattern, response)[0]
        except:
            pass
            
        code_blocks = response.split("'''")
        if len(code_blocks) >= 3:
            return self.ordering_str, code_blocks[1].strip()
            
        if response.lower().startswith("minizinc"):
            return self.ordering_str, response[len("minizinc"):].strip()
            
        if "% minizinc" in response.lower():
            return self.ordering_str, response[response.lower().find("% minizinc"):].strip()
            
        return self.ordering_str, response.strip()

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
                # Split by comma, then by colon
                assignments = {}
                for item in assignments_str.split(','):
                    item = item.strip()
                    if not item:
                        continue
                    if ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to convert to int, otherwise keep as string
                        try:
                            assignments[key] = int(value)
                        except ValueError:
                            assignments[key] = value
                solutions.append({'assignments': {variable_name: assignments}})
        return solutions