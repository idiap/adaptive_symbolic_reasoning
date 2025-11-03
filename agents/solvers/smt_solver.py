# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.base import Scratchpad
from typing import Any, Dict, List
import subprocess
import tempfile
import os
import logging
import sys
from .abstract import AbstractSolver
from base.logger import setup_logging
log = setup_logging()

class SMTSolver(AbstractSolver):
    """Agent for solving SMT problems using Z3."""

    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Dict = None, predecessors: List[str] = None, problem_to_solve: List[str] = None) -> None:
        super().__init__(name, goal, generator, prompt_dict, predecessors, problem_to_solve)

    def __call__(self, memory: Scratchpad, **model_args) -> Dict[str, Any]:
        """Solve an SMT problem."""
        
        try:
            # Read data with first predecessor's name suffix
            first_pred = self.predecessors[0]
            problem_queue = memory.read(f"smt_problem_queue_{first_pred}") or []
            if len(problem_queue) == 0:
                problem = memory.read(f'smt_input_{first_pred}')
                if not isinstance(problem, list):
                    problem = [problem]
                problem_queue = problem
            if not problem_queue:
                # No more problems to solve
                return {"success": False, "error": "Queue empty"}
            
            # Take the last problem from the queue
            problem = problem_queue.pop()
            memory.write(f"smt_problem_queue_{self.name}", problem_queue)
            
            # Check if problem is already formalized
            if problem.get('formalized', False):
                code = problem['smt_code']
            else:
                # Formalize the problem
                problem_desc = problem['problem']
                code = self._formalize(problem_desc, **model_args)
                
            # Solve the problem
            for attempt in range(self.max_attempts):
                log.info("Attempt %d/%d", attempt + 1, self.max_attempts)
                success, error_msg, solutions = self._run_solver(code)

                if success:
                    log.info("Z3 ran successfully. Found %d solutions", len(solutions) if solutions else 0)
                    if solutions:
                        result = {
                            'ori_answer': solutions[0]['assignments'],
                            'parsed_answer': 'A', # A) True; B) False
                            'success': True,
                            'error': None,
                            'is_satisfiable': 'A', # A) True; B) False
                            'code': code,
                            # 'rule_index': problem.get('rule_index'),
                            # 'rule_natural_language': problem.get('rule_natural_language'),
                            # 'instance_natural_description': problem.get('instance_natural_description')
                        }
                    else:
                        log.warning("No solutions found in successful run")
                        result = {
                            'ori_answer': solutions,
                            'parsed_answer': 'B', # A) True; B) False
                            'success': True,
                            'error': None,
                            'is_satisfiable': 'B', # A) True; B) False
                            'code': code,
                            # 'rule_index': problem.get('rule_index'),
                            # 'rule_natural_language': problem.get('rule_natural_language'),
                            # 'instance_natural_description': problem.get('instance_natural_description')
                        }
                    
                    # Add result to results queue  
                    results_queue = memory.read("smt_results_queue") or []
                    results_queue.append(result)
                    memory.write("smt_results_queue", results_queue)
                    memory.write(f"result_{self.get_problem_key()}", result)
                    
                    return result
                else:
                    log.warning("Z3 error: %s", error_msg)
                    
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
            log.error("Failed to solve SMT: %s", str(e))
            raise ValueError(f"Failed to solve SMT: {str(e)}")

    def _formalize(self, problem: str, output_anyway: bool = False, **model_args) -> str:
        """Convert problem to SMT-LIB format."""
        if 'response_format' in model_args:
            del model_args['response_format']
        try:
            del self.llm.client.model_kwargs['response_format']
        except:
            pass
        response = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['generate smt from prompt'],
            model_args = model_args,
            premise=problem
        )
        return self._extract_code_from_response(response)
    
    def _fix_syntax_error(self, code: str, error_msg: str, **model_args) -> str:
        """Fix SMT-LIB syntax errors."""
        if 'response_format' in model_args:
            del model_args['response_format']
        fixed_code = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine smt error'],
            model_args = model_args,
            code='"""'+code+'"""',
            error="Z3 Error",
            error_detail=error_msg
        )
        return self._extract_code_from_response(fixed_code)

    def _extract_code_from_response(self, response: str) -> str:
        """Extract SMT-LIB code from LLM response."""
        code = response
        # First try to extract code from triple quotes
        code_blocks = response.split("'''")
        if len(code_blocks) >= 3:
            code = code_blocks[1].strip()

        # Remove lines that are just keywords
        keywords = {"smt-lib", "smt", "smtlib"}
        lines = code.splitlines()
        filtered_lines = [
            line for line in lines
            if line.strip().lower() not in keywords
        ]
        return "\n".join(filtered_lines).strip()
    

    def _run_solver(self, code: str) -> tuple[bool, str, list[Dict[str, Any]]]:
        """Run Z3 solver on SMT-LIB code."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run Z3
                log.info("Running Z3 command: z3 -smt2 %s", temp_file)
                result = subprocess.run(
                    ['z3', '-smt2', temp_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if result.stderr:
                    return False, result.stderr, []
                
                # Parse output
                output = result.stdout.lower()
                if "unsat" in output:
                    return True, "", []
                elif "sat" in output:
                    # Extract model if available
                    model = {}
                    for line in result.stdout.split('\n'):
                        if line.startswith('(define-fun'):
                            # Parse variable assignments
                            parts = line.strip('()').split()
                            if len(parts) >= 4:
                                var_name = parts[1]
                                value = parts[-1]
                                model[var_name] = value
                    return True, "", [{'assignments': model}]
                else:
                    return False, "Unexpected Z3 output - neither sat nor unsat found", []
                
            except subprocess.CalledProcessError as e:
                log.error("Z3 subprocess error: %s", e)
                log.error("Z3 stdout: %s", e.stdout)
                log.error("Z3 stderr: %s", e.stderr)
                return False, e.stdout + e.stderr, []
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            log.error("Exception in _run_solver: %s", str(e))
            return False, str(e), []

    
