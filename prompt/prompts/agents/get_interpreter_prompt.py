# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}


prompt_dict['relation_extractor'] = {
    "goal": "Extract every qualitative effect relation from the text.",
    "system_prompt":  "You are an expert in molecular pathways. Extract every qualitative effect "
                            "relation from the text in JSON array form with keys: source, relation, target, context.",
    "user_prompt": "Input: [INPUT]\n\n"
}

prompt_dict['interpreter'] = {
    "goal": "Give LLM-based interpretation of the final activation states after perturbation.",
    "system_prompt": "Given the final activation states of proteins after perturbation, "
                    "write a concise natural-language summary describing how initial changes "
                    "propagate through the pathway and which proteins are most affected. "
                    "Provide context-specific interpretation.",
    "user_prompt": "Input: [INPUT]\n\n"
}
