# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from agents.meta_agents.planner import Planner

def get_plans(data, generator, portfolio):
    planner = Planner(generator = generator)
    plan, memory = planner(data, portfolio)[0]
    return plan, memory
