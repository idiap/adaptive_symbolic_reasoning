# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import xml.etree.ElementTree as ET
import json
import random
from helpers.clinical_trials_loader import get_eligibility_criteria
import re
import os

TOPICS_XML = "data/TREC/topics2021.xml"
QRELS_TXT = "data/TREC/qrels2021.txt"
OUT_JSON = "data/TREC/trec_patient_trial_testset.json"
FIXED_TOPICS_XML = "data/TREC/topics_2021_fixed.xml"
MAX_EXAMPLES = 150

def fix_xml_ampersands(xml_path, fixed_path):
    """Fix unescaped ampersands in XML and write to a new file."""
    with open(xml_path, "r") as f:
        xml = f.read()
    # Replace & not followed by a word and a semicolon (i.e., not an entity)
    xml_fixed = re.sub(r'&(?![a-zA-Z]+;)', '&amp;', xml)
    with open(fixed_path, "w") as f:
        f.write(xml_fixed)
    print(f"Wrote fixed XML to {fixed_path}")

def parse_topics(xml_path):
    """Parse topics XML and return dict patient_id -> description."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    patients = {}
    for topic in root.findall("topic"):
        pid = topic.attrib["number"]
        desc = topic.text.strip()
        patients[pid] = desc
    return patients

def parse_qrels(qrels_path):
    """Parse qrels and return list of (patient_id, trial_id, label)."""
    pairs = []
    with open(qrels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            patient_id, _, trial_id, label = parts
            if label in {"1", "2"}:
                pairs.append((patient_id, trial_id, int(label)))
    return pairs

def main():
    # Fix XML if needed
    fix_xml_ampersands(TOPICS_XML, FIXED_TOPICS_XML)
    patients = parse_topics(FIXED_TOPICS_XML)
    pairs = parse_qrels(QRELS_TXT)

    # Group by patient and label
    patient_trials = {}
    for patient_id, trial_id, label in pairs:
        if patient_id not in patient_trials:
            patient_trials[patient_id] = {1: [], 2: []}
        patient_trials[patient_id][label].append(trial_id)

    selected = []
    for patient_id in sorted(patients.keys(), key=lambda x: int(x)):
        patient_desc = patients.get(patient_id)
        if not patient_desc:
            continue
        trials = patient_trials.get(patient_id, {1: [], 2: []})
        for label in [1, 2]:
            if trials[label]:
                trial_id = random.choice(trials[label])
                trial_desc = get_eligibility_criteria(trial_id)
                if not trial_desc:
                    continue
                selected.append({
                    "patient_id": patient_id,
                    "patient_description": patient_desc,
                    "trial_id": trial_id,
                    "trial_description": trial_desc,
                    "match": label == 2
                })

    with open(OUT_JSON, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved {len(selected)} datapoints to {OUT_JSON}")

if __name__ == "__main__":
    main()