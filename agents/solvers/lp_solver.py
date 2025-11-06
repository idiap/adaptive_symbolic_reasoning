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
import sys

from .abstract import AbstractSolver
from pyke import knowledge_engine

# Set up logging
from base.logger import setup_logging
log = setup_logging()



class LPSolver(AbstractSolver):
    def __init__(self, name: str, goal: str, generator: Any, prompt_dict: Dict = None, expect_json = True, predecessors: List[str] = None, problem_to_solve: List[str] = None):
        super().__init__(name, goal, generator, prompt_dict, predecessors, problem_to_solve)
        self.expect_json = expect_json
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.cache_dir = config['cache_dir']['pyke_cache']
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        os.system(f'rm -rf {self.cache_dir}/*')

        self.f_index = 0

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

            memory.write(f"intermediate_form_{self.name}", intermediate_form)

            # Then try to solve it
            for attempt in range(self.max_attempts):
                history_critique_output.append(intermediate_form)

                log.info("Attempt %d/%d", attempt + 1, self.max_attempts)
                self._initialize()
                success, error_msg, solutions = self._run_solver()
                memory.write(f"critique_outputs_{self.name}", solutions)
                if success:
                    log.info("Pyke solver ran successfully.")
                    if solutions:
                        solver_label = self._get_final_result(memory, use_agent = True, response_format={"type": "json_object"})
                        
                        result = {
                            'ori_answer': solutions,
                            'parsed_answer': solver_label,
                            'intermediate_form': intermediate_form,
                            'Rules': str(self.Rules),
                            'Facts': str(self.Facts)
                        }
                        memory.write(f"result_{self.get_problem_key()}", result)

                        return result
                    else:
                        log.warning("No solutions found in successful run")
                else:
                    log.warning("Pyke solver error: %s", error_msg)
                    
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
                prompt_name = self.prompt_dict['get pyke form'],
                model_args = model_args,
                premise = premise,
                conclusion = hypothesis,
                include_explanations = False
            )
        else:
            inference_result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name = self.prompt_dict['get pyke form'],
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
        else:
            self._save_formalised_kb()
        
        return response

    def _parse_forward_rule(self, rule):
        statements = rule.split('>>>')
        pyke_rule = None  # Initialize to handle empty rules
        
        for idx in range(len(statements)-1):
            premise, conclusion = statements[idx], statements[idx+1]
            self.f_index += 1
            premise = premise.strip()
            # split the premise into multiple facts if needed
            premise = premise.split('&&')
            premise_list = [p.strip() for p in premise]

            conclusion = conclusion.strip()
            # split the conclusion into multiple facts if needed
            conclusion = conclusion.split('&&')
            conclusion_list = [c.strip() for c in conclusion]

            # create the Pyke rule
            pyke_rule = f'''fact{self.f_index}\n\tforeach'''
            for p in premise_list:
                pyke_rule += f'''\n\t\tfacts.{p}'''
            pyke_rule += f'''\n\tassert'''
            for c in conclusion_list:
                pyke_rule += f'''\n\t\tfacts.{c}'''
        
        if pyke_rule is None:
            log.warning("Malformed rule (no >>> separator): %s", rule)
            return ""  # Return empty string for malformed rules
        
        return pyke_rule
    
    def _create_fact_file(self, facts):
        with open(os.path.join(self.cache_dir, 'facts.kfb'), 'w') as f:
            for fact in facts:
                # check for invalid facts
                if not fact.find('$x') >= 0:
                    f.write(fact + '\n')

    def _create_rule_file(self, rules):
        pyke_rules = []
        for idx, rule in enumerate(rules):
            pyke_rules.append(self._parse_forward_rule(rule))

        with open(os.path.join(self.cache_dir, 'rules.krb'), 'w') as f:
            f.write('\n\n'.join(pyke_rules))

    def _save_formalised_kb(self) -> None:
        self._create_fact_file(self.Facts)
        self._create_rule_file(self.Rules)

    def _fix_syntax_error(self, memory: Scratchpad, error_msg: str, **model_args) -> str:
        if 'response_format' in model_args:
            del model_args['response_format']
        first_pred = self.predecessors[0]
        hypotheses = memory.read(f"hypothesis_{first_pred}") or None
        premise = memory.read(f"premise_{first_pred}") or None
        formulations = memory.read(f"intermediate_form_{self.name}") or None
        refined = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine pyke error'],
            model_args = model_args,
            premise=premise,
            hypotheses=hypotheses,
            formulations = formulations,
            code='"""'+'\n'.join(self.Facts) + '\n\n' + '\n'.join(self.Rules)+'"""', 
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

    def _find_first_letter_position(self, s):
        for index, char in enumerate(s):
            if char.isalpha():
                return index
        return -1 

    def _parse_segment(self, program_str, key_phrase):
        remain_program_str, segment = program_str.split(key_phrase)
        segment_list = segment.strip().split('\n')
        for i in range(len(segment_list)):
            segment_list[i] = segment_list[i].split(':::')[0].strip()
            first_letter_index = self._find_first_letter_position(segment_list[i])
            segment_list[i] = segment_list[i][first_letter_index:]
        return remain_program_str, segment_list

    def _validate_program(self):
        if not self.Rules is None and not self.Facts is None:
            if not self.Rules[0] == '' and not self.Facts[0] == '':
                return True, ""
        # try to fix the program
        tmp_rules = []
        tmp_facts = []
        statements = self.Facts if self.Facts is not None else self.Rules
        if statements is None:
            return False, "The program is empty"
        
        for fact in statements:
            if fact.find('>>>') >= 0: # this is a rule
                tmp_rules.append(fact)
            else:
                tmp_facts.append(fact)
        self.Rules = tmp_rules
        self.Facts = tmp_facts
        return True, "Program error but tried to fix the program."

    def _extract_code_from_response(self, program_str: str) -> str:
        try: 
            keywords = ['Query:', 'Rules:', 'Facts:', 'Predicates:']
            for keyword in keywords:
                try:
                    program_str, segment_list = self._parse_segment(program_str, keyword)
                    setattr(self, keyword[:-1], segment_list)
                except:
                    setattr(self, keyword[:-1], None)
            processing_flag, error_str = self._validate_program()
            return processing_flag, error_str
        except Exception as e:
            return False, str(e)
        
    def _check_specific_predicate(self, subject_name, predicate_name, engine):
        results = []
        with engine.prove_goal(f'facts.{predicate_name}({subject_name}, $label)') as gen:
            for vars, plan in gen:
                results.append(vars['label'])

        with engine.prove_goal(f'rules.{predicate_name}({subject_name}, $label)') as gen:
            for vars, plan in gen:
                results.append(vars['label'])

        if len(results) == 1:
            return results[0]
        elif len(results) == 2:
            return results[0] and results[1]
        elif len(results) == 0:
            return None
        
    def _parse_query(self, query):
        pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
        match = re.match(pattern, query)
        if match:
            function_name = match.group(1)
            arg1 = match.group(2)
            arg2 = match.group(3)
            arg2 = True if arg2 == 'True' else False
            return function_name, arg1, arg2
        else:
            raise ValueError(f'Invalid query: {query}')

    def _run_solver(self) -> Tuple[bool, str, List[Dict[str, Any]]]:
        try:
            if os.path.exists('./compiled_krb'):
                os.system(f'rm -rf ./compiled_krb')

            engine = knowledge_engine.engine(self.cache_dir)

            engine.reset()
            engine.activate('rules')
            engine.get_kb('facts')
            predicate, subject, value_to_check = self._parse_query(self.Query[0])
            value_after_infer = self._check_specific_predicate(subject, predicate, engine)

            if value_after_infer is None:
                return True, '', 'Unknown'
            elif value_after_infer == value_to_check:
                return True, '', 'True'
            else:
                return True, '', 'False'

        except Exception as e:
            log.error("Error running Pyke: %s", str(e))
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
                    prompt_name = self.prompt_dict['match pyke solution'],
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

    def _initialize(self):
        if os.path.exists('./compiled_krb'):
            os.system(f'rm -rf ./compiled_krb')

        for mod in list(sys.modules.keys()):
            if mod.startswith('compiled_krb'):
                del sys.modules[mod]
