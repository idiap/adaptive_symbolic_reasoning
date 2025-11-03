# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import json
import yaml
import random
import os

def get_options_and_answer(target_scores):
    LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
    options = list(target_scores.keys())
    correct_idx = [i for i, k in enumerate(options) if target_scores[k] == 1]
    if not correct_idx:
        raise ValueError("No correct answer in target_scores")
    correct_letter = LETTERS[correct_idx[0]]
    return options, correct_letter, correct_idx[0]

data_names = ["pronto_qa","proof_writer","folio","logical_deduction","trec"]
formulations = {
    "pronto_qa": "lp",
    "proof_writer": "lp",
    "folio": "FOL",
    "logical_deduction": "CSP",
    "trec": "SMT"
}

reasoning_types = {
    "pronto_qa": "Deductive Reasoning",
    "proof_writer": "Deductive Reasoning",
    "folio": "First-order Logic",
    "logical_deduction": "Constraint Satisfaction",
    "trec": "Analytic Reasoning"
}


data_templates = {
    "pronto_qa": "STATEMENT:\n{premise}\n\nQUESTION:\n{conclusion}\n\n{options}",
    "proof_writer": "STATEMENT:\n{premise}\n\nQUESTION:\n{conclusion}\n\n{options}",
    "folio": "STATEMENT:\n{premise}\n\n\nQuestion:\n{conclusion}\n\n{options}",
    "logical_deduction": "STATEMENT:\n{input}\n\nWhich of the following is true?\n\n{options}",
    "trec": "You get a trial and a patient and have to say if there is a match:\n\n"+"TRIAL: {trial_description}\n\nPATIENT: {patient_description}\n\nDoes the patient match the trial?\nA) True\nB) False"
}

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if os.path.exists('data/mixed') is False:
    os.makedirs('data/mixed')

result_data = []
result_subset = []
for data_name in data_names:
    data_dir = config.get('data_dir', {}).get(data_name+'_ori')
    with open(data_dir, "r") as f:
        data = json.load(f)
    
    print('Merging %d samples from %s' % (len(data), data_name))

    template = data_templates[data_name]
    generated_data = []
    for idx, datum in enumerate(data):
        temp_record = {
            "id": formulations[data_name] + '_' + str(datum.get('id', idx)),
            "source": data_name,
            "type": reasoning_types[data_name],
        }
        if data_name in ['pronto_qa', 'proof_writer', 'folio']:
            options_str = '\n'.join(datum['options'])
            problem_str = template.format(premise=datum['context'], conclusion=datum['question'], options = options_str)
            answer_str = datum['answer']

        elif data_name in ['logical_deduction']:
            problem_str = template.format(input=datum['context'], options = '\n'.join(datum['options']).replace(')','.'))
            answer_str = datum['answer']
            
        elif data_name == 'trec':
            problem_str = template.format(trial_description=datum['trial_description'], patient_description=datum['patient_description'])
            answer_str = 'A' if datum['match'] is True else 'B'

        else:
            raise ValueError(f"Data format for {data_name} is not recognized.")
        temp_record.update({
            "problem": problem_str,
            "answer": answer_str
        })

        generated_data.append(temp_record)
    generated_subset = random.sample(generated_data, min(50, len(generated_data)))
    with open(f"data/mixed/sep_{data_name}.json", "w") as f:
        json.dump(generated_data, f, indent=2)
    result_data.extend(generated_data)
    result_subset.extend(generated_subset)



random.shuffle(result_data)
with open("data/mixed/mixed_data.json", "w") as f:
    json.dump(result_data, f, indent=2)

random.shuffle(result_subset)
with open("data/mixed/mixed_subset.json", "w") as f:
    json.dump(result_subset, f, indent=2)

        

        
