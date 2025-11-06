# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

########################################################################################################
# Note:
# Some implementations are referenced from the following link:
# https://github.com/teacherpeterpan/Logic-LLM
########################################################################################################

from agents.base import Scratchpad
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import yaml
import logging
from nltk.inference.prover9 import *

from .abstract import AbstractSolver
from agents.critiques.prover9_parser import Prover9_FOL_Formula
from agents.critiques.prover9_formula import FOL_Formula

# Set up logging
from base.logger import setup_logging
log = setup_logging()



class FOLSolver(AbstractSolver):
    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Dict = None, expect_json = True, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, generator, prompt_dict, predecessors, problem_to_solve)
        self.expect_json = expect_json
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.cache_dir = config['cache_dir']['prover9_cache']
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        os.system(f'rm -rf {self.cache_dir}/*')

    def __call__(self, memory: Scratchpad, **model_args) -> Dict[str, Any]:
        try:
            # Read data with first predecessor's name suffix
            first_pred = self.predecessors[0]
            explanations = memory.read(f"explanation_{first_pred}") or None
            hypothesis = memory.read(f"hypothesis_{first_pred}") or None
            premise: str| None = memory.read(f"premise_{first_pred}") or None
            if premise is None: # In case the premise and explanations are concatenated
                premise = memory.read(f'statements_{first_pred}')
                hypothesis = memory.read(f'goal_{first_pred}')
                explanations = premise[-1]
                premise = ' '.join(premise[:-1])
            
            if premise is None or hypothesis is None:
                raise ValueError("Incomplete input")
            
            history_critique_output = memory.read("history_critique_output") or []

            # First formalize the problem
            intermediate_form = self._formalize(
                explanations=explanations,
                hypothesis=hypothesis,
                premise=premise,
                **model_args
            )

            # Then try to solve it
            for attempt in range(self.max_attempts):
                history_critique_output.append(intermediate_form)

                log.info("Attempt %d/%d", attempt + 1, self.max_attempts)
                success, error_msg, solutions = self._run_solver()
                memory.write(f"critique_outputs_{self.name}", solutions)
                if success:
                    log.info("Prover9 solver ran successfully.")
                    if solutions:
                        solver_label = self._get_final_result(memory, use_agent = True, response_format={"type": "json_object"})
                        
                        result = {
                            'ori_answer': solutions,
                            'parsed_answer': solver_label,
                            'intermediate_form': intermediate_form,
                            'premises': str(self.prover9_premises),
                            'conclusion': str(self.prover9_conclusion)
                        }
                        memory.write(f"result_{self.get_problem_key()}", result)

                        return result
                    else:
                        log.warning("No solutions found in successful run")
                else:
                    log.warning("Prover9 solver error: %s", error_msg)
                    
                if attempt < self.max_attempts - 1:
                    try:
                        log.info("Attempting to fix error...")
                        intermediate_form = self._fix_syntax_error(memory, error_msg, **model_args) 
                        log.info("Code fixed, trying again...")
                    except Exception as e:
                        log.error("Error during fix attempt: %s", str(e))
                        if attempt == self.max_attempts - 1:
                            raise ValueError(f"Failed to solve after {self.max_attempts} attempts: {str(e)}")
                        continue
                    
            raise ValueError(f"Failed to solve after {self.max_attempts} attempts")
        except Exception as e:
            log.error("Failed to solve logic programing problem: %s", str(e))
            raise ValueError(f"Failed to solve logic programing problem: {str(e)}")
        

    def _format_result(self, result):
            result_strs = result.split('\n')
            formatted_strs = []
            for line in result_strs:
                cleaned_line = re.sub(r'^\s*(\d+\.\s*|\-\s*)', '', line)
                formatted_strs.append(cleaned_line)
            return '\n'.join(formatted_strs)

    def _formalize(self, explanations: str,
                    hypothesis: str,
                    premise: Optional[str],
                    output_anyway: bool = False,
                    **model_args) -> str:
        def format_sentences(sentences):
            formatted = ''
            sentence_list = re.split(r'[.\n]\s*', sentences)
            for order, sentence in enumerate(sentence_list, start=1):
                if sentence.strip():
                    formatted += (
                        f"Explanatory Sentence {order}: {sentence.strip()}\n"
                        )
            return formatted
        
        if 'response_format' in model_args:
            del model_args['response_format']
        
        model_args.update({'max_tokens':2048})
        try:
            del self.llm.client.model_kwargs['response_format']
        except:
            pass
        if explanations is None or len(explanations) == 0:
            inference_result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name = self.prompt_dict['get prover9 form'],
                model_args = model_args,
                premise = premise,
                conclusion = hypothesis,
                include_explanations = False
            )
        else:
            inference_result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name = self.prompt_dict['get prover9 form'],
                model_args = model_args,
                premise = premise,
                explanations=format_sentences(explanations),
                conclusion = hypothesis
            )
        
        response = self._format_result(inference_result)

        with open(os.path.join(self.cache_dir, 'intermediate_form.txt'), 'w') as f:
            f.write(response)
        
        flag, error_str = self._extract_code_from_response(response)

        if not flag and not output_anyway:
            raise ValueError(error_str)
        
        return response

    def _fix_syntax_error(self, memory: Scratchpad, error_msg: str, **model_args) -> str:
        if 'response_format' in model_args:
            del model_args['response_format']
        first_pred = self.predecessors[0]
        hypotheses = memory.read(f"hypothesis_{first_pred}") or None
        premise = memory.read(f"premise_{first_pred}") or None
        formulations = memory.read(f"intermediate_form_{self.name}") or None
        refined = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine prover9 error'],
            model_args = model_args,
            premise=premise,
            hypotheses=hypotheses,
            formulations = formulations,
            code='"""'+'\n'.join(self.prover9_premises) + '\n\n' + self.prover9_conclusion+'"""', 
            error=error_msg
        )
        
        if not refined:
            raise ValueError("LLM returned empty response for refinement")
        
        response = self._format_result(refined)

        with open(os.path.join(self.cache_dir, 'intermediate_form.txt'), 'w') as f:
            f.write(response)
        
        flag, error_str = self._extract_code_from_response(response)

        if not flag:
            raise ValueError(error_str)
        
        return response

    def _extract_code_from_response(self, response: str) -> str:
        try: 
            # Split the string into premises and conclusion
            if 'Conclusions:' not in response:
                response = response.replace('Conclusion:', 'Conclusions:')
            premises_string = response.split("Conclusions:")[0].split("Premises:")[1].strip()
            conclusion_string = response.split("Conclusions:")[1].strip()

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split('\n')
            conclusion = conclusion_string.strip().split('\n')

            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_conclusion = conclusion[0].split(':::')[0].strip()

            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False, "Invalid premise formula"
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False, "Invalid conclusion formula"
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True, ""
        except Exception as e:
            return False, str(e)

    def _run_solver(self) -> Tuple[bool, str, List[Dict[str, Any]]]:
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 600

            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()

            if result:
                return True, '', 'True'
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                # negation_result = prover.prove(negated_goal, assumptions)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover.prove()
                if negation_result:
                    return True, '', 'False'
                else:
                    return True, '', 'Unknown'
        except Exception as e:
            log.error("Error running Prover9: %s", str(e))
            return False, str(e), ''


    def _get_final_result(self, memory, use_agent = True, **model_args):
        critique_output = memory.read(f"critique_outputs_{self.name}")
        if isinstance(critique_output, list):
            critique_output = critique_output[0]
        critique_output = str(critique_output)
        if critique_output == 'None':
            critique_output = 'Unknown'
        first_pred = self.predecessors[0]
        options = memory.read(f"options_{first_pred}") or None

        if options is None:
            return critique_output['result']
        else:
            if use_agent:
                reply = self.llm.generate(
                    model_prompt_dir = 'helpers',
                    prompt_name = self.prompt_dict['match prover9 solution'],
                    model_args = model_args,
                    options = options,
                    generated_text = critique_output
                )
                
                label = reply.get('label')
            else:
                if critique_output == 'True':
                    label = 'A'
                elif critique_output == 'False':
                    label = 'B'
                else:
                    label = 'C'

            memory.write(f'solver_label_{self.name}', label)

            return label