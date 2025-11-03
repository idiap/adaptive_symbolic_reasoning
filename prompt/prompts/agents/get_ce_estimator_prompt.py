# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}

prompt_dict['causal_structure_extractor'] = {
    "goal": "Parse the argument.",
    "system_prompt":"You are a helpful assistant to extract and parse causal factors and relations in the text. [GOAL] Respond only with json. Keys: "
                    "'treatment', 'outcome', 'confounders', 'causal_statements'. "
                    "The 'causal_statements' key should be a list of causal statements in the form of '<Cause> increases|decreases <Effect> by <Δ>% (p=<pvalue>)'. ",
    "user_prompt": "Statement: [STATEMENT]\n\n"
}

prompt_dict['probability_extractor'] = {
    "goal": "Extract the probabilities from the text.",
    "system_prompt":  "You are an expert data scientist. Extract numeric probabilities only "
                                "and respond as json. Schema A: {p_t, p_y1_t1, p_y1_t0}. "
                                "Schema B: {p_t, p_c, p_y1_t1_c, p_y1_t0_c}.",
    "user_prompt": "Input: [INPUT]\n\n"
}