# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}

prompt_dict['scheme_catalog'] = {
    # ------------------------------------------------------------------
    # SCHEME CATALOG  –  our local “ground truth”
    # ------------------------------------------------------------------
    "argument from expert opinion": {
        "family": "Authority",
        "definition": "A claim is accepted because a qualified expert says it is true.",
        "roles": ["Expert statement", "Expert credibility", "Domain", "Consistency"],
        "critical_questions": [
            "Is E an expert in domain D?",
            "Is E trustworthy?",
            "Did E really assert A?",
            "Is A consistent with what other experts assert?",
            "Is A consistent with known evidence?"
        ]
    },
    "practical reasoning": {
        "family": "Practical",
        "definition": "From an agent’s goals and available means we infer a course of action.",
        "roles": ["Goal", "Means", "Side effects", "Value"],
        "critical_questions": [
            "Will the action achieve the goal?",
            "Are there better alternative means?",
            "Are the side-effects acceptable?",
            "Is the goal itself desirable?"
        ]
    },
    "argument from cause to effect": {
        "family": "Causal",
        "definition": "Because C is a sufficient (or necessary) cause of E, C implies E.",
        "roles": ["Cause", "Causal law", "Effect"],
        "critical_questions": [
            "Is the causal relation well established?",
            "Could other factors block or counteract the cause?",
            "Is the causal link reversible (effect-to-cause)?"
        ]
    },
    "argument from analogy": {
        "family": "Comparative",
        "definition": "Object A has property P because it is similar to object B which has P.",
        "roles": ["Base case", "Target case", "Shared properties", "Difference"],
        "critical_questions": [
            "Are A and B similar in all relevant respects?",
            "Are there relevant dissimilarities?",
            "Is there a more appropriate analogy?"
        ]
    }
}


prompt_dict['argument_structure_parser'] = {
    "goal": "Parse the argument.",
    "system_prompt": "You are an argument-analysis assistant. "
        "Return JSON with keys 'premises' (array) and 'conclusion' (string).",
    "user_prompt": "Extract the argument structure:\n\n[ARGUMENT]"
}

prompt_dict['argument_scheme_classifier'] = {
    "goal": "Identify the argumentation scheme of the given argument.",
    "system_prompt": "Return ONLY the name of the best matching Walton scheme.",
    "user_prompt": f"Premises: [PREMISE]\nConclusion: [CONCLUSION]\n\n"
                f"Below are the name and the explanation of each available scheme: \n[SCHEME]"
}

prompt_dict['critical_question_generator'] = {
    "goal": "Prefer the catalog’s CQs; fall back to the model if we have no entry.",
    "system_prompt": "Return the critical questions for this Walton scheme (JSON array).",
    "user_prompt": "Scheme: [SCHEME]"
}

prompt_dict['premise_conclusion_connector'] = {
    "goal": "Map nodes & edges.  If we have a role template in the catalog we give it to the LLM to encourage consistent labelling.",
    "system_prompt": "You are an argument-mapping assistant. "
                     "Output JSON with keys 'nodes' and 'edges' exactly as described: "
                     "nodes need id, text, role; edges need from, to, type (supports|attacks). "
                     "Include the conclusion as node id 'c'.",
    "user_prompt": "Scheme: [SCHEME]\n[HINT]\n"
              f"Premises: [PREMISE]\nConclusion: [CONCLUSION]"
}

prompt_dict['walton_arg_analyser'] = {
    'goal': "Adds an internal classification taxonomy for Walton’s schemes."
}
