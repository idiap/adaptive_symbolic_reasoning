# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import json
import random
import os
from typing import List, Dict, Any

def shuffle_dataset(data: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Shuffle dataset order"""
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    return shuffled_data

def merge_problems(problem_group: List[Dict[str, Any]], group_id: int) -> Dict[str, Any]:
    # Build merged problem text
    merged_problem_parts = []
    for i, problem in enumerate(problem_group, 1):
        problem_text = problem['problem'].strip()
        merged_problem_parts.append(f"Q{i}: {problem_text}")
    
    # Add unified instruction
    merged_problem = "Answer the following questions one by one.\n\n"
    merged_problem += "\n\n".join(merged_problem_parts)
    
    # Merge metadata
    merged_sources = [p['source'] for p in problem_group]
    merged_types = [p['type'] for p in problem_group] 
    merged_answers = [p['answer'] for p in problem_group]
    merged_ids = [p['id'] for p in problem_group]
    
    # Create merged data entry
    merged_entry = {
        "id": f"multi_problem_{group_id}",
        "source": merged_sources,
        "type": merged_types,
        "system_message": "Solve the following multiple reasoning problems. Each problem may require different reasoning approaches (First-order Logic, Constraint Satisfaction, Deductive Reasoning, etc.). Answer each question individually.",
        "problem": merged_problem,
        "answer": merged_answers,
        "original_ids": merged_ids,
        "num_problems": len(problem_group)
    }
    
    return merged_entry

def create_multi_problem_dataset(problems_per_group: int = 3, shuffle_seed: int = 42) -> None:
    """
    Create multi-problem dataset
    
    Args:
        problems_per_group: Number of problems to merge per group
        shuffle_seed: Random seed
    """
    
    input_file = "data/mixed/mixed_data.json"
    
    # Create output directory
    output_dir = "data/concat"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/concat_{problems_per_group}.json"
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    print(f"Loaded {len(original_data)} problems")
    
    print(f"Shuffling dataset with seed {shuffle_seed}...")
    shuffled_data = shuffle_dataset(original_data, shuffle_seed)
    
    print(f"Creating groups of {problems_per_group} problems...")
    merged_dataset = []
    
    # Group and merge data
    for i in range(0, len(shuffled_data), problems_per_group):
        group = shuffled_data[i:min(i + problems_per_group, len(shuffled_data))]
            
        merged_problem = merge_problems(group, len(merged_dataset) + 1)
        merged_dataset.append(merged_problem)
    
    print(f"Created {len(merged_dataset)} merged problems")
    
    # Save results
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
    
    print("Dataset creation completed!")
    
    # Output statistics
    print("\n=== Statistics ===")
    print(f"Original problems: {len(original_data)}")
    print(f"Merged problems: {len(merged_dataset)}")
    print(f"Problems per group: {problems_per_group}")
    print(f"Utilization rate: {len(merged_dataset) * problems_per_group / len(original_data) * 100:.1f}%")

if __name__ == "__main__":
    # Create dataset with 5 problems merged by default
    create_multi_problem_dataset(problems_per_group = 3)