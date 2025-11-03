# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}


prompt_dict['structure_extractor'] = {
    "goal": "Parses effect statements via LLM if API key is set, else via regex fallback.",
    "system_prompt": "You are a statistician. From the text, extract all statements of form: "
                            "'<Cause> increases|decreases <Effect> by <Δ>% (p=<pvalue>)'. "
                            "If pvalue is not given, fill in the bracket with p=0.05. "
                            "Return a JSON array 'struct', which includes a list of dict objects having keys: "
                            "cause[str], effect[str], delta[float], p_value[float], baseline[float]."
                            "The data type of each keyword is specified in square brackets; strictly follow these type specifications.",
    "user_prompt": "Input: [INPUT]\n\n"
}
