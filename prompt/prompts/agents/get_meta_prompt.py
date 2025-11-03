# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}


prompt_dict['planner'] = {
    "system_prompt": "Design a plan that uses the minimal number of agents necessary to achieve the goals. "
                    "You may be given one or multiple problems to solve, each with a unique problem_id. "
                    "Select agents only from the provided list. Output a JSON object describing the plan."
                    "Requirements:\n"
                    "1. Use the MINIMAL number of agents needed to complete all tasks.\n"
                    "2. The output MUST be a JSON object with exactly two keys:\n"
                    "  • 'agents': an array of selected agent names (strings), including any problem_ids that serve as starting points\n"
                    "  • 'edges': an array of [source, target] pairs (both strings) showing execution order.\n"
                    "3. For multi-problem scenarios, start the execution flow from the respective problem_ids.\n"
                    "4. The plan MUST end with the special control marker <PLAN_END>.\n"
                    "5. IMPORTANT: If you need to use the same agent type for multiple different problems, distinguish them by adding ':' followed by a sequence number. "
                    "For example, if you need two CSP solvers for different problems, use 'csp_solver:1' and 'csp_solver:2'.\n"
                    "6. You can use the same agent for multiple problems if appropriate, but ensure proper sequencing.\n"
                    "Example for multiple problems:\n"
                    "If given 'QID [ques_1]: CSP scheduling problem', 'QID [ques_2]: FOL reasoning task' and 'QID [ques_3]: CSP allocation problem', "
                    "your output might include agents: ['ques_1', 'ques_2', 'ques_3', 'csp_solver:1', 'fol_solver:1', 'csp_solver:2', '<PLAN_END>'] "
                    "with edges: [['ques_1', 'csp_solver:1'], ['ques_2', 'fol_solver:1'], ['ques_3', 'csp_solver:2'], ['csp_solver:1', '<PLAN_END>'], ['fol_solver:1', '<PLAN_END>'], ['csp_solver:2', '<PLAN_END>']].\n"
                    "7. Respond with **only valid JSON**, without any explanations, markdown formatting, or code fences. "
                    "Do not wrap the output in ```json or any other delimiters. Return pure JSON.",
    "user_prompt": f"Here is the goal: [GOAL]\n\nProblems to solve: [PROBLEMS]\n\nCurrent agents: \n[PORTFOLIO]"
}

prompt_dict['planner_local'] = {
    "system_prompt": "Design a plan that uses the minimal number of agents necessary to achieve the goals. "
                    "You may be given one or multiple problems to solve, each with a unique problem_id. "
                    "Select agents only from the provided list. Output your reasoning process, which should include a JSON object describing the plan."
                    "\n\nCurrent agents: \n[PORTFOLIO]"
                    "\n\nRequirements:\n"
                    "1. Use the MINIMAL number of agents needed to complete all tasks.\n"
                    "2. The output MUST be a JSON object with exactly two keys:\n"
                    "  • 'agents': an array of selected agent names (strings), including any problem_ids that serve as starting points\n"
                    "  • 'edges': an array of [source, target] pairs (both strings) showing execution order.\n"
                    "3. For both single-problem and multi-problem scenarios, start the execution flow from the respective problem_ids.\n"
                    "4. The plan MUST end with the special control marker <PLAN_END>.\n"
                    "5. IMPORTANT: If you need to use the same agent type for multiple different problems, distinguish them by adding ':' followed by a sequence number. "
                    "For example, if you need two CSP solvers for different problems, use 'csp_solver:1' and 'csp_solver:2'.\n"
                    "Example for single problem:\n"
                    "If given 'QID [ques_1]: SMT scheduling problem', "
                    "then 'ques_1' is the problem id, and your output might include agents: ['ques_1', 'smt_solver:1',  '<PLAN_END>'] "
                    "with edges: [['ques_1', 'smt_solver:1'], ['smt_solver:1', '<PLAN_END>']].\n"
                    "Example for multiple problems:\n"
                    "If given 'QID [ques_1]: CSP scheduling problem', 'QID [ques_2]: FOL reasoning task' and 'QID [ques_3]: CSP allocation problem', "
                    "your output might include agents: ['ques_1', 'ques_2', 'ques_3', 'csp_solver:1', 'fol_solver:1', 'csp_solver:2', '<PLAN_END>'] "
                    "with edges: [['ques_1', 'csp_solver:1'], ['ques_2', 'fol_solver:1'], ['ques_3', 'csp_solver:2'], ['csp_solver:1', '<PLAN_END>'], ['fol_solver:1', '<PLAN_END>'], ['csp_solver:2', '<PLAN_END>']].\n"
                    "6. Think step by step, and respond your answer in **valid JSON**.",
    "user_prompt": f"Here is the goal: [GOAL]\n\nProblems to solve: [PROBLEMS]"
}