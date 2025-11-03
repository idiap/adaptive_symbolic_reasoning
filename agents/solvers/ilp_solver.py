# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.base import Scratchpad
from typing import Any, Dict, List, Tuple, Optional
import subprocess
import tempfile
import os
import logging
from .abstract import AbstractSolver
from base.logger import setup_logging
log = setup_logging()

class ILPSolver(AbstractSolver):
    """Agent for solving ILP problems using Popper."""

    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Dict = None, popper_path: Optional[str] = None, predecessors: List[str] = None, problem_to_solve: List[str] = None) -> None:
        super().__init__(name, goal, generator, prompt_dict, predecessors, problem_to_solve)
        self.popper_path = popper_path if popper_path else 'popper-ilp'
        self.found_rules = []
        self.bias_code = None
        self.bk_code = None
        self.examples_code = None

    def __call__(self, memory: Scratchpad, **model_args) -> Dict[str, Any]:
        """Solve an ILP problem."""
        try:
            # Read data with first predecessor's name suffix
            first_pred = self.predecessors[0]
            data = memory.read(f"ilp_input_{first_pred}")
            
            if not isinstance(data, dict) or 'problem' not in data:
                raise ValueError("Invalid input format")
            
            if not data['problem']:
                raise ValueError("Problem description is empty")

            # First formalize the problem
            formalization = self._formalize(data['problem'], **model_args)
            self.bias_code = formalization['bias']
            self.bk_code = formalization['background']
            self.examples_code = formalization['examples']
                    
            # Then try to solve it
            for attempt in range(self.max_attempts):
                log.info("Attempt %d/%d", attempt + 1, self.max_attempts)
                success, error_msg, rules = self._run_solver()
                
                if success:
                    log.info("Popper ran successfully. Found %d rules", len(rules) if rules else 0)
                    if rules:
                        result = {
                            'success': True,
                            'error': None,
                            'rules': rules,
                            'code': {
                                'bias': self.bias_code,
                                'background': self.bk_code,
                                'examples': self.examples_code
                            }
                        }
                        memory.write(f"result_{self.get_problem_key()}", result)
                        return result
                    else:
                        log.warning("No rules found in successful run")
                        result = {
                            'success': False,
                            'error': 'No rules were found',
                            'code': {
                                'bias': self.bias_code,
                                'background': self.bk_code,
                                'examples': self.examples_code
                            }
                        }
                        memory.write(f"result_{self.get_problem_key()}", result)
                        return result
                else:
                    log.warning("Popper error: %s", error_msg)
                    
                if attempt < self.max_attempts - 1:
                    try:
                        log.info("Attempting to fix error...")
                        formalization = self._fix_syntax_error(
                            {
                                'bias': self.bias_code,
                                'background': self.bk_code,
                                'examples': self.examples_code
                            },
                            error_msg,
                            **model_args
                        )
                        self.bias_code = formalization['bias']
                        self.bk_code = formalization['background']
                        self.examples_code = formalization['examples']
                        log.info("Code fixed, trying again...")
                    except Exception as e:
                        log.error("Error during fix attempt: %s", str(e))
                        if attempt == self.max_attempts - 1:
                            raise ValueError(f"Failed to solve after {self.max_attempts} attempts: {str(e)}")
                        continue
                    
            raise ValueError(f"Failed to solve after {self.max_attempts} attempts")
            
        except Exception as e:
            log.error("Failed to solve ILP: %s", str(e))
            raise ValueError(f"Failed to solve ILP: {str(e)}")

    def _formalize(self, problem: str, output_anyway: bool = False, **model_args) -> Dict[str, str]:
        """Convert problem to ILP format (bias.pl, bk.pl, exs.pl)."""
        if 'response_format' in model_args:
            del model_args['response_format']
        try:
            del self.llm.client.model_kwargs['response_format']
        except:
            pass
        response = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['get_ilp_form_prompt'],
            model_args = model_args,
            problem=problem
        )
        return self._extract_ilp_code_from_response(response)
    
    def _fix_syntax_error(self, code: Dict[str, str], error_msg: str, **model_args) -> Dict[str, str]:
        """Fix ILP syntax errors."""
        # Combine the code into a single string with comments
        current_code = f"% bk.pl\n{code['background']}\n\n"
        current_code += f"% exs.pl\n{code['examples']}\n\n"
        current_code += f"% bias.pl- defines predicates that can be used\n{code['bias']}"

        if 'response_format' in model_args:
            del model_args['response_format']
        
        log.info("Calling LLM for refinement...")
        response = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine_ilp_form_prompt'],
            model_args = model_args,
            current_code=current_code,
            error=error_msg
        )
        return self._extract_ilp_code_from_response(response)

    def _extract_ilp_code_from_response(self, response: str) -> Dict[str, str]:
        """Extract ILP code (bias.pl, bk.pl, exs.pl) from LLM response."""
        # Split response into sections based on comments
        lines = response.split('\n')
        bias_lines = []
        bk_lines = []
        examples_lines = []
        current_section = None
        
        for line in lines:
            if '% bias.pl' in line:
                current_section = 'bias'
                continue
            elif '% bk.pl' in line:
                current_section = 'bk'
                continue
            elif '% exs.pl' in line:
                current_section = 'examples'
                continue
            elif line.strip().startswith('%'):
                continue
                
            if current_section == 'bias':
                bias_lines.append(line)
            elif current_section == 'bk':
                bk_lines.append(line)
            elif current_section == 'examples':
                examples_lines.append(line)
        
        return {
            'bias': '\n'.join(bias_lines).strip(),
            'background': '\n'.join(bk_lines).strip(),
            'examples': '\n'.join(examples_lines).strip()
        }

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response - required by AbstractSolver."""
        # For ILP, we need to return a combined string representation
        # This method is mainly for compatibility with the abstract class
        if response is None:
            raise ValueError("LLM response is None")
        
        # Try to extract ILP code and return a combined representation
        try:
            ilp_code = self._extract_ilp_code_from_response(response)
            combined = f"Bias:\n{ilp_code['bias']}\n\nBackground:\n{ilp_code['background']}\n\nExamples:\n{ilp_code['examples']}"
            return combined
        except Exception:
            # Fallback to returning the raw response
            return response.strip()

    def _run_solver(self) -> Tuple[bool, str, List[str]]:
        """Run Popper solver on ILP files."""
        try:
            # Create temporary directory for Popper files
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Write files
                with open(os.path.join(tmp_dir, 'bias.pl'), 'w') as f:
                    f.write(self.bias_code)
                with open(os.path.join(tmp_dir, 'bk.pl'), 'w') as f:
                    f.write(self.bk_code)
                with open(os.path.join(tmp_dir, 'exs.pl'), 'w') as f:
                    f.write(self.examples_code)
                
                # Run Popper
                popper_cmd = [
                    self.popper_path,
                    '--max-vars', '6',
                    '--timeout', '60',
                    tmp_dir
                ]
                
                result = subprocess.run(
                    popper_cmd,
                    capture_output=True,
                    text=True,
                    timeout=70
                )

                log.debug("Popper output: %s", result.stdout)
                
                if result.stderr:
                    log.debug("Popper errors: %s", result.stderr)
                
                if result.returncode != 0:
                    return False, f"Popper failed with return code: {result.returncode}", []
                
                # Parse rules from output
                rules = []
                for line in result.stdout.split('\n'):
                    if ':-' in line:  # Basic rule detection
                        rules.append(line.strip())
                
                return True, "", rules
                
        except subprocess.TimeoutExpired:
            return False, "Popper timed out", []
        except subprocess.CalledProcessError as e:
            return False, f"Popper failed: {e.stderr}", []
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", [] 
