# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.solvers.lp_solver import LPSolver
from agents.solvers.fol_solver import FOLSolver
from agents.solvers.csp_solver import CSPSolver
from agents.solvers.smt_solver import SMTSolver
from agents.solvers.ilp_solver import ILPSolver
from agents.solvers.LLM_solver import LLMSolver


from agents.base import LLMToolAgent, Portfolio, BaseAgent, SpecialControlMarker
from base.logger import setup_logging
log = setup_logging()

import os
import yaml

import os
import subprocess



with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# MiniZinc
minizinc_path = config['agent_config']['minizinc_path']
os.environ['PATH'] = os.environ['PATH']+ ':'+f'{minizinc_path}/bin:'
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + f':/usr/lib:/usr/local/lib:{minizinc_path}/lib:'
os.environ['QT_PLUGIN_PATH'] = os.environ.get('QT_PLUGIN_PATH', '') + f':{minizinc_path}/plugins:'

# Prover9
os.environ['PROVER9'] = './libs/Prover9/bin'

def register_agent(portfolio: Portfolio, agent: BaseAgent, name: str, goal: str, prompt_dict: dict = {}) -> None:
    if name in portfolio.keys():
        raise ValueError(f"Agent name collision: {name}")
    portfolio[name] = {
        'goal': goal,
        'agent': agent,
        'prompt_dict':prompt_dict
        }
    log.info("Registered agent '%s'", name)



def initialize_agents():
    # Seed portfolio
    portfolio: Portfolio = {}

    register_agent(portfolio, 
        SpecialControlMarker,
        name = "<PLAN_START>", 
        goal = "Special control marker, indicating the start of the whole working plan.")
    register_agent(portfolio, 
        SpecialControlMarker,
        name = "<PLAN_END>", 
        goal = "Special control marker, indicating the end of the whole working plan.")

    # Solvers
    register_agent(portfolio,
        LPSolver,
        name = "lp_solver",
        goal = "Determine whether a hypothesis can be derived from the given facts and rules using logical deduction..",
        prompt_dict={
            'get pyke form': 'get_pyke_forms_prompt_solver.txt',
            'refine pyke error': 'refine_pyke_form_prompt.txt',
            'match pyke solution': 'match_options.txt'
        })
    register_agent(portfolio,
        FOLSolver,
        name = "fol_solver",
        goal = "A theorem prover specialized for problems expressed in first-order logic (FOL). It is capable of handling complex reasoning tasks that involve quantifiers, predicates, and relationships among multiple entities.",
        prompt_dict={
            'get prover9 form': 'get_prover9_forms_prompt_solver.txt',
            'refine prover9 error': 'refine_prover9_form_prompt.txt',
            'match prover9 solution': 'match_options.txt'
        })
    register_agent(portfolio,
        CSPSolver,
        name = "csp_solver",
        goal = "A solver tailored for constraint satisfaction problems (CSP) where the goal is to assign values to variables within finite domains such that all explicit or implicit constraints are satisfied. It focuses on configuration, scheduling, and allocation tasks.",
        prompt_dict={
            'generate minizinc from prompt': 'get_csp_form_from_prompt.txt',
            'refine minizinc error': 'refine_csp_form_prompt.txt'
        })
    register_agent(portfolio,
        SMTSolver,
        name = "smt_solver",
        goal = "A boolean satisfiability (SAT) solver designed to verify whether all logical constraints in a system are simultaneously satisfied. It is particularly effective for analytical reasoning tasks that involve checking whether a given configuration fulfills complex sets of conditions.",
        prompt_dict={
            'generate smt from prompt': 'get_smt_form_from_prompt_fewshot_complex.txt',
            'refine smt error': 'refine_smt_form_prompt.txt'
        })
    register_agent(portfolio,
        ILPSolver,
        name = "ilp_solver",
        goal = "Agent for solving inductive logic programming (ILP) problems using Popper.",
        prompt_dict={
            'get_ilp_form_prompt': 'get_ilp_form_prompt.txt',
            'refine_ilp_form_prompt': 'refine_ilp_form_prompt.txt'
        }) 
    register_agent(portfolio, 
        LLMSolver,
        name = "epistemic_solver", 
        goal = "Determining what is true from mixed or conflicting evidence within the premise. Includes resolution of contradictions between sources, preferring objective measurements (labs, imaging) over opinions, and establishing diagnostic status from an evidence hierarchy.",
        prompt_dict={
            'get_solution': 'Epistemic_solver.txt',
            'get_refinement': 'Epistemic_refiner.txt',
            'get_verification': 'Epistemic_verifier_merged.txt',
        }
        )
    register_agent(portfolio, 
        LLMSolver,
        name = "risk_solver", 
        goal = "Risk ranking or comparison (highest risk, safer, dangerous), weighing severity against frequency, expected-harm reasoning, and hazards not ruled out by the premise.",
        prompt_dict={
            'get_solution': 'Risk_solver.txt',
            'get_refinement': 'Risk_refiner.txt',
            'get_verification': 'Risk_verifier_merged.txt'
        }
        )
    register_agent(portfolio, 
        LLMSolver,
        name = "compositional_solver",
        goal = "Joint constraints over drug–dose–units–schedule–diagnosis–patient factors (age, sex, renal/hepatic function, comorbidities) and co-therapy. Includes dosing bounds, indications, exclusions, and concurrency rules.",
        prompt_dict={
            'get_solution': 'Composition_solver.txt',
            'get_refinement': 'Composition_refiner.txt',
            'get_verification': 'Composition_verifier_merged.txt'
        }
        ) 
    register_agent(portfolio, 
        LLMSolver,
        name = "causal_solver",
        goal = "Statements making causal claims “effect of T on Y” (e.g., cause, lead to, improve, reduce, accelerate;). May include or omit an interventional contrast or comparator to verify.",
        prompt_dict={
            'get_solution': 'Causal_solver.txt',
            'get_refinement': 'Causal_refiner.txt',
            'get_verification': 'Causal_verifier_merged.txt'
        }
        ) 

    return portfolio
