# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}


EVIDENCE_TYPES = [
    "theory", "simulation", "interventional study", "observational study",
    "survey", "meta-analysis", "expert opinion", "qualitative", "anecdote",
]

prompt_dict['proposition_extractor'] = {
    "goal": "Extract every distinct proposition from the text.",
    "system_prompt": "You are an evidence analyst. Extract *every distinct proposition* from the text. [GOAL]\n"
            "Return strict json array where each element has: claim[str], evidence_excerpt[str], evidence_type[str], "
            "strength_features (population_size[int], p_value[float], effect_size[float], replication[bool]), strength_score[float].\n"
            "The data type of each keyword is specified in square brackets; strictly follow these type specifications.\n"
            "Evidence types: " + ", ".join(EVIDENCE_TYPES) + ".",
    "user_prompt": "Input: [INPUT]\n\n"
}