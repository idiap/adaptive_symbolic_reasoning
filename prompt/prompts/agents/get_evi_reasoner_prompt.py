# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}


prompt_dict['hypotheses_identifier'] = {
    "goal": "Extract from the text a set of mutually exclusive and semantically related hypotheses relevant to the goal",
    "system_prompt": "You are a helpful data-fusion assistant specialized in Dempster–Shafer evidence theory."
            "You are given a decision goal and a set of independent natural-language evidence statements."
            "Your task is to extract a set of mutually exclusive and semantically related hypotheses relevant to the goal."
            "These hypotheses should be general enough to allow belief assignment from multiple statements, and should avoid being overly fragmented or too specific."
            "The hypotheses should form an exhaustive and logically coherent frame of discernment (e.g., contrasting perspectives on the same issue)."
            "Respond with ONE JSON object (no markdown fences, no extra text)."
            "The JSON object must contain a single key \"frame\", whose value is a list of hypotheses.",
    "user_prompt": "Goal: [GOAL]\nSTATEMENTS:\n[STATEMENTS]"
}


prompt_dict['dempster_shafer_reasoner'] = {
    "goal": "Assign masses to the hypotheses based on the evidence statements",
    "system_prompt": "You are a helpful data-fusion assistant.\n"
            "The user gives you:\n"
            "  • A CLOSED list of mutually-exclusive hypotheses (Θ).\n"
            "  • One or more evidence statements (independent sources).\n\n"
            "Respond with ONE JSON object (no markdown fences, no extra text).\n"
            "The object must have a key \"masses\", whose value is an array. "
            "Each element of that array must contain:\n"
            "  focal : list of hypotheses from Θ (use [] to denote Θ itself)\n"
            "  mass  : float, 0 < mass ≤ 1.\n"
            "The sum of all masses must be ≤ 1.\n",
    "user_prompt": "Θ = [FRAME]\nSTATEMENTS:\n[STATEMENTS]"
}