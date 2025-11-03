# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import json
import yaml
import os
import shutil
from agents.meta_agents.planner  import TracePersister, Planner
from agents.meta_agents.initializer import initialize_agents
from base.logger import setup_logging
from base.error_collector import setup_error_collection, get_error_collector
log = setup_logging()

# Setup error collection to capture errors from across the system
error_collector = setup_error_collection()


def evaluating_agent(test_data_name, llm, max_examples = -1, backbone_type = 'api'):
    ################### initialization ###################
    # initialize all available agents
    planner = Planner(generator = llm)

    # set result path
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # load configuration
    result_path = config.get('cache_dir', {}).get('result_dir')
    test_data_name_str = test_data_name.replace('_subset', '')
    path_str = f"results_{test_data_name_str}"
    path_str += '_agent'
    path_str += f'_{backbone_type}'
    path_str += '.json'
    result_path_agent = os.path.join(result_path, path_str)
    log.info('*'*20+f'Saving to {result_path_agent}'+'*'*20)

    ################### load the test data ###################
    data_dir = config.get('data_dir', {}).get(test_data_name)
    with open(data_dir, "r") as f:
        test_data = json.load(f)
    if max_examples != -1:
        test_data = test_data[:max_examples]
    
    # check if test data has 'id' attribute, if not, create one
    id_check = test_data[0].get('id', None)
    if id_check is None:
        for idx, entry in enumerate(test_data):
            entry['id'] = idx

    # check if there are already evaluated cases, if so, get their ids
    evaluated_ids = []
    results = []
    if os.path.exists(result_path_agent):
        with open(result_path_agent, "r") as f:
            results = json.load(f)
        evaluated_ids = {r["id"] for r in results}
    
    ################### testing ###################
    # test all samples
    for idx, entry in enumerate(test_data):
        # check if already evaluated
        if entry['id'] in evaluated_ids:
            log.info(f"Skipping already evaluated case with ID {entry['id']}")
            continue

        # Set session ID for error tracking
        session_id = f"case_{entry['id']}"
        error_collector.set_session_id(session_id)
        
        # Clear previous errors for this session
        error_collector.clear_errors()


        problem_types = []
        try:
            if not isinstance(entry['answer'], list):
                entry['answer'] = [entry['answer'], ]
            # parse data for the solver and design the plan
            plan, memory, problem_ids = planner(entry)
            plan = plan[0]

            for problem_id in problem_ids:
                problem_types.append(memory.read(f'problem_type_{problem_id}'))

            # execute the plan
            plan.execute(memory, TracePersister())
            ori_answers = []
            parsed_answers = []
            for problem_id in problem_ids:
                result = memory.read(f"result_{problem_id}")
                if result is None: 
                    result = memory.read(f"result_no_problem") or {}
                ori_answers.append(result.get('ori_answer', None))
                parsed_answers.append(result.get('parsed_answer', None))

            is_all_correct = True
            is_partial_correct = []
            for parsed_answer, gt in zip(parsed_answers, entry['answer']):
                if parsed_answer != gt:
                    is_partial_correct.append(False)
                    is_all_correct = False
                else:
                    is_partial_correct.append(True)
            temp_res = {
                "id": entry['id'],
                "pred_problem_type":problem_types,
                "evaluation_partial": is_partial_correct,
                "evaluation": is_all_correct,
                "prediction": parsed_answers,
                "solver_output": ori_answers,
                "groundtruth": entry['answer'],
                "problem": entry['problem'],
                "plan_agents": list(plan.agents.keys()),
                "plan_dag": '\n'.join([edge.source + ' -> ' + edge.target for edge in plan.edges]),
                "type": entry['type'],
            }
            if "original_ids" in entry:
                temp_res["original_ids"] = entry['original_ids']
            log.info(f'Agentic Solver id ' + entry['id'] + f': predicted: {parsed_answer}, groundtruth: '+ str(entry['answer']) +f', partial evaluation: {is_partial_correct}, overall evaluation: {is_all_correct}')
        except Exception as e:
            import traceback

            # Capture full traceback as a formatted string
            traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

            # Log the full traceback
            log.error(f'Error on id {entry["id"]}:\n{traceback_str}')

            temp_res = {
                "id": entry['id'],
                "pred_problem_type":problem_types,
                "evaluation_partial": None,
                "evaluation": False,
                "prediction": None,
                "solver_output": None,
                "groundtruth": entry['answer'],
                "problem": entry['problem'],
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback_str,
                "type": entry['type']
            }
            if "original_ids" in entry:
                temp_res["original_ids"] = entry['original_ids']
            try:
                temp_res['plan_agents'] = list(plan.agents.keys())
                temp_res['plan_dag'] = '\n'.join([edge.source + ' -> ' + edge.target for edge in plan.edges]),
            except:
                pass
            error_summary = error_collector.get_error_summary(session_id)
            temp_res['error_info'] = error_summary['errors']

            if error_summary['total_errors'] > 0:
                log.info(f"Case {entry['id']}: Collected {error_summary['total_errors']} errors during processing")
        
        results.append(temp_res)
        # Save after each case to avoid losing progress
        with open(result_path_agent, "w") as f:
            json.dump(results, f, indent=2)

    # Final accuracy
    correct = sum(1 for r in results if r["evaluation"])
    total = len(results)
    log.info(f"Agentic Solver Accuracy: {correct}/{total} ({100.0 * correct / total:.2f}%)")
    return results
