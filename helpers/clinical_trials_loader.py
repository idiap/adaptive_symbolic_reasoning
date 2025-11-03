# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import requests
import random
from typing import List, Optional

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def get_trial_ids(n: int, condition: str) -> List[str]:
    """
    Fetch a list of up to N trial IDs for a given condition.

    Args:
        n: Number of trial IDs to return.
        condition: Condition to search for.

    Returns:
        List of NCT IDs (strings).
    """
    params = {
        "query.cond": condition,
        "pageSize": 100,  # Max allowed per page
        "fields": "NCTId"
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Error fetching trial IDs: HTTP {response.status_code}")
        return []
    data = response.json()
    trials = data.get('studies', [])
    ids = [
        trial['protocolSection']['identificationModule']['nctId']
        for trial in trials
        if 'protocolSection' in trial and
           'identificationModule' in trial['protocolSection'] and
           'nctId' in trial['protocolSection']['identificationModule']
    ]
    if not ids:
        print("No trial IDs found for the given condition.")
        return []
    return random.sample(ids, min(n, len(ids)))

def get_eligibility_criteria(nct_id: str) -> Optional[str]:
    """
    Fetch eligibility criteria for a given trial ID.

    Args:
        nct_id: The NCT ID of the trial.

    Returns:
        Eligibility criteria as a string, or None if not found.
    """
    detail_url = f"{BASE_URL}/{nct_id}"
    response = requests.get(detail_url)
    if response.status_code != 200:
        print(f"Error fetching details for {nct_id}: HTTP {response.status_code}")
        return None
    data = response.json()
    eligibility = data.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria')
    if not eligibility:
        print(f"No eligibility criteria found for {nct_id}.")
    return eligibility