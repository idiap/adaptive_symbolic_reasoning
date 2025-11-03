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

from typing import Optional
import re
import yaml
import os

from nltk.inference.prover9 import *
from .abstract import FormalisationModel, HardCritiqueModel
from .prover9_parser import Prover9_FOL_Formula
from .prover9_formula import FOL_Formula




class Prover9Formaliser(FormalisationModel):
    def __init__(self, name, goal, generator, prompt_dict: Optional[dict] = None):
        super().__init__(name, goal, generator, prompt_dict)
        self.f_index = 0
        self.cache_dir = './temp/formalisation_prover9'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _parse_logic_program(self, logic_program):
        try:        
            # Split the string into premises and conclusion
            premises_string = logic_program.split("Conclusions:")[0].split("Premises:")[1].strip()
            conclusion_string = logic_program.split("Conclusions:")[1].strip()

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
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except:
            return False

    def _get_prover9_proof(self,cache_dir) -> str:
        goal = Expression.fromstring(self.prover9_conclusion)
        assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
        timeout = 10

        prover = Prover9Command(goal, assumptions, timeout=timeout)
        result = prover.prove()

        if result:
            return 'True'
        else:
            # If Prover9 fails to prove, we differentiate between False and Unknown
            # by running Prover9 with the negation of the goal
            negated_goal = NegatedExpression(goal)
            # negation_result = prover.prove(negated_goal, assumptions)
            prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
            negation_result = prover.prove()
            if negation_result:
                return 'False'
            else:
                return 'Unknown'

    def _get_intermediate_form(self, premise: str, explanations: str,
                    hypothesis: str, **model_args) -> str:

        def format_sentences(sentences):
            formatted = ''
            sentence_list = re.split(r'[.\n]\s*', sentences)
            for order, sentence in enumerate(sentence_list, start=1):
                if sentence.strip():
                    formatted += (
                        f"Explanatory Sentence {order}: {sentence.strip()}\n"
                        )
            return formatted

        def format_result(result):
            result_strs = result.split('\n')
            formatted_strs = []
            for line in result_strs:
                cleaned_line = re.sub(r'^\s*(\d+\.\s*|\-\s*)', '', line)
                formatted_strs.append(cleaned_line)
            return '\n'.join(formatted_strs)
        
        model_args.update({'max_tokens':2048})
        
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
        
        return format_result(inference_result)
        
    def _get_prover9_form(self, premise: str, explanations: str,
                    hypothesis: str,
                    **model_args) -> str:
        intermediate_form = self._get_intermediate_form(
            premise,
            explanations,
            hypothesis,
            **model_args
        )
        with open(os.path.join(self.cache_dir, 'intermediate_form.txt'), 'w') as f:
            f.write(intermediate_form)
        flag = self._parse_logic_program(intermediate_form)
        return flag
    
    def formalise(self, theory_name: str, premise: str,
                  explanation: str, hypothesis: str,
                  logical_form: str = 'event-based semantics',
                  **model_args) -> str:
        if logical_form == 'event-based semantics':
            self.processing_flag = self._get_prover9_form(
                premise,
                explanation,
                hypothesis,
                **model_args
            )
            return self.processing_flag
    
    def save_formalised_kb(self):
        pass

        


class Prover9Critique(HardCritiqueModel):
    def __init__(self, name, goal, generator, 
                 theory_name: Optional[str] = 'example',
                 prompt_dict: Optional[dict] = None):
        super().__init__(name, goal, generator, prompt_dict)
        if prompt_dict is None:
            prompt_dict = {
                'get prover9 form':
                    'get_prover9_forms_prompt_solver.txt',
            }
        self.prover9_name = 'test'
        self.verbose = True
        self.options = None
        self.watchdog_timeout = 60 #150
        self.code = None
        self.theory_name = theory_name
        self.prompt_dict = prompt_dict
        self.formaliser = Prover9Formaliser(name, goal, generator, prompt_dict)
        self.prover9_dir = self._get_prover9_dir()

        if not os.path.exists(self.prover9_dir):
            os.makedirs(self.prover9_dir)
            
        os.system(f'rm -rf {self.prover9_dir}/*')

    def _get_prover9_dir(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config['cache_dir']['prover9_cache']


    def _get_formalisation(self, explanation: str,
                           hypothesis: str,
                           premise: Optional[str],
                           theory_name: Optional[str],
                           **model_args):
        
        # get prover9 code from input natural language sentences
        self.processing_flag = self.formaliser.formalise(
            theory_name, premise,
            explanation, hypothesis,
            logical_form='event-based semantics',
            **model_args
        )

        return self.processing_flag
    

    def _get_prover9_syntax_output(self, theory_name: str,
                                    explanation: str,
                                    hypothesis: str,
                                    premise: Optional[str] = None,
                                    **model_args) -> bool:
        # formalise the nl into prover9 theory
        processing_flag = self._get_formalisation(
            explanation=explanation,
            hypothesis=hypothesis,
            premise=premise,
            theory_name=theory_name,
            **model_args
        )
        if not processing_flag:
            return False, None
        else:
            logical_information = self.formaliser._get_prover9_proof(self.prover9_dir)
            return processing_flag, logical_information

    def critique(self, iteration_number: int,
                 explanation: str,
                 hypothesis: str,
                 premise: Optional[str], 
                 **model_args):
        theory_name = f'{self.theory_name}_{str(iteration_number)}'
        error_code = ''
        semantic_validity = True
        syntactic_validity = True
        solving_time = 0
        logical_information = ''
        critique_outputs = {}

        (syntactic_validity,
            logical_information) = self._get_prover9_syntax_output(
            theory_name=theory_name,
            explanation=explanation,
            hypothesis=hypothesis,
            premise=premise,
            **model_args
        )
        # if has syntax error, return directly
        if not syntactic_validity:
            semantic_validity = False
        critique_outputs['semantic validity'] = semantic_validity
        critique_outputs['syntactic validity'] = syntactic_validity
        critique_outputs['error code'] = error_code.strip()
        critique_outputs['solving time'] = solving_time
        critique_outputs['logical information'] = logical_information
        return critique_outputs

    def shutdown(self):
        pass
