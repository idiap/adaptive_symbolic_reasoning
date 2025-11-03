# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

prompt_dict = {}

prompt_dict['narrator'] = {
    "goal": "Generates a clear narrative comparing fitted polynomial relationships based on described visual features.",
    "system_prompt": "You are a data analyst. Given visual feature descriptions and a list of "
                    "symbolic formulas for different polynomial degrees, produce a clear narrative "
                    "explaining the plot and comparing the fitted relationships.",
    "user_prompt": "Features: [FEATURES]. \n"
                "Fitted formulas: [FORMULAS]"
}