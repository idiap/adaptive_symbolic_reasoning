# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from pyswip import Prolog
import tempfile
import subprocess
import json
import os
import shutil
import re
import logging
from agents.solvers.ilp_solver import ILPSolver
from agents.base import Scratchpad

log = logging.getLogger("agentic_sdk")

class DrugInteractionKnowledgeBase(ILPSolver):
    """Knowledge base for ILP-based reasoning about drug interactions"""
    
    def __init__(self, 
                 name: str,
                 goal: str,
                 popper_path: Optional[str] = None,
                 bias_file: Optional[str] = None,
                 bk_file: Optional[str] = None,
                 examples_file: Optional[str] = None):
        """
        Initialize the drug interaction knowledge base
        
        Args:
            name: Name of the agent
            goal: Goal of the agent
            popper_path: Path to Popper installation. If None, assumes Popper is in PATH
            bias_file: Path to bias.pl file.
            bk_file: Optional path to background knowledge file
            examples_file: Optional path to examples file
        """
        super().__init__(name, goal, popper_path)
        
        log.info("Initializing DrugInteractionKnowledgeBase with name: %s, goal: %s", name, goal)
        
        # Add KB-specific prompts to the existing prompt dictionary
        kb_prompts = {
            'formalize_fact': 'formalize_drug_fact.txt',
            'formalize_interaction': 'formalize_drug_interaction.txt'
        }
        
        # Append KB prompts to parent's prompt dictionary
        self.prompt_dict.update(kb_prompts)
        
        log.info("Initialized with prompts: %s", self.prompt_dict)
        
        # Initialize storage
        self.allowed_predicates: Dict[str, Dict] = {}  # predicate -> {type: str, direction: tuple}
        self.background_facts: Dict[str, Set[Tuple[str, str]]] = {}
        self.positive_interactions: Set[Tuple[str, str]] = set()
        self.negative_interactions: Set[Tuple[str, str]] = set()
        
        # Track all known entities
        self.known_proteins: Set[str] = set()
        self.known_drugs: Set[str] = set()

        # Verify required files exist
        if not bias_file:
            raise ValueError("Bias file is required for Popper to work")
        if not examples_file:
            raise ValueError("Examples file is required for Popper to work")
        if not bk_file:
            raise ValueError("Background knowledge file is required for Popper to work")

        # Load files in the correct order
        log.info("Loading bias file: %s", bias_file)
        self.load_bias_file(bias_file)  # First load bias to initialize predicates
        log.info("Loading examples file: %s", examples_file)
        self.load_examples(examples_file)  # Then load examples to get initial drug names
        log.info("Loading background knowledge file: %s", bk_file)
        self.load_background_knowledge(bk_file)  # Finally load background knowledge to add more drug names
        
        log.info("Initialized with %d known proteins and %d known drugs", 
                len(self.known_proteins), len(self.known_drugs))

    def add_fact_from_nl(self, description: str, memory: Optional[Scratchpad] = None) -> bool:
        """
        Add a drug-target relationship from natural language description
        
        Args:
            description: Natural language description of the relationship
            memory: Optional shared memory for storing results
            
        Returns:
            bool: True if fact was successfully added
        """
        # Call LLM to formalize the fact
        response = self.llm.generate(
            model_prompt_dir='kb',
            prompt_name=self.prompt_dict['formalize_fact'],
            description=description,
            known_proteins=json.dumps(sorted(list(self.known_proteins))),
            known_drugs=json.dumps(sorted(list(self.known_drugs))),
            known_predicates=json.dumps(sorted(list(self.allowed_predicates.keys())))
        )
        
        # Check for error message from LLM
        if response.strip().startswith('ERROR:'):
            log.error("LLM Error: %s", response.strip())
            return False
        
        try:
            # Parse response to get predicate and arguments
            match = re.search(r'([a-z_]+)\(([^,]+),([^)]+)\)', response)
            if not match:
                log.error("Could not parse LLM response: %s", response)
                return False
                
            pred, arg1, arg2 = match.groups()
            
            # Verify predicate is known
            if pred not in self.allowed_predicates:
                log.error("Unknown predicate: %s", pred)
                return False
            
            # Handle different predicate types
            if pred == 'target':
                protein, drug = arg1, arg2
                if protein not in self.known_proteins:
                    log.error("Unknown protein name: %s", protein)
                    return False
                if drug not in self.known_drugs:
                    log.error("Unknown drug name: %s", drug)
                    return False
            elif pred.startswith('target'):
                drug, protein = arg1, arg2
                if protein not in self.known_proteins:
                    log.error("Unknown protein name: %s", protein)
                    return False
                if drug not in self.known_drugs:
                    log.error("Unknown drug name: %s", drug)
                    return False
            
            log.info("Adding to background knowledge (bk.pl): %s(%s,%s).", pred, arg1, arg2)
            
            # Add the fact
            return self.add_fact(pred, arg1, arg2)
            
        except Exception as e:
            log.error("Error processing fact: %s", str(e))
            return False

    def add_interaction_from_nl(self, description: str, memory: Optional[Scratchpad] = None) -> bool:
        """
        Add a drug-drug interaction from natural language description
        
        Args:
            description: Natural language description of the interaction
            memory: Optional shared memory for storing results
            
        Returns:
            bool: True if interaction was successfully added
        """
        # Call LLM to formalize the interaction
        response = self.llm.generate(
            model_prompt_dir='kb',
            prompt_name=self.prompt_dict['formalize_interaction'],
            description=description,
            known_drugs=json.dumps(sorted(list(self.known_drugs))),
            known_predicates=json.dumps(sorted(list(self.allowed_predicates.keys())))
        )
        
        # Check for error message from LLM
        if response.strip().startswith('ERROR:'):
            log.error("LLM Error: %s", response.strip())
            return False
        
        try:
            # Parse response to get drug names and interaction type
            match = re.search(r'(pos|neg)\(interacts\(([^,]+),([^)]+)\)\)', response)
            if not match:
                log.error("Could not parse LLM response: %s", response)
                return False
                
            interaction_type, drug1, drug2 = match.groups()
            
            # Double check that both drug names are known
            if drug1 not in self.known_drugs:
                log.error("Unknown drug name: %s", drug1)
                return False
            if drug2 not in self.known_drugs:
                log.error("Unknown drug name: %s", drug2)
                return False
            
            log.info("Adding to examples (exs.pl): %s(interacts(%s,%s)).", interaction_type, drug1, drug2)
            
            # Add the interaction
            self.add_interaction(drug1, drug2, positive=(interaction_type == 'pos'))
            return True
            
        except Exception as e:
            log.error("Error processing interaction: %s", str(e))
            return False

    def load_bias_file(self, bias_file: str):
        """Load predicate declarations and their constraints from the bias file"""
        self.bias_file = bias_file
        
        if not os.path.exists(self.bias_file):
            raise FileNotFoundError(f"Bias file not found: {self.bias_file}")
            
        # Clear existing predicates
        self.allowed_predicates.clear()
            
        # Read file to process predicates
        with open(self.bias_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%') or line.startswith('#'):
                    continue
                
                if line.startswith('body_pred(') or line.startswith('head_pred('):
                    pred = line[line.index('(')+1:line.index(',')]
                    self.allowed_predicates[pred] = {'type': None, 'direction': None}
                    if pred not in self.background_facts:
                        self.background_facts[pred] = set()
                elif line.startswith('type('):
                    content = line[line.index('(')+1:line.rindex(')')]
                    pred, types = content.split(',', 1)
                    if pred in self.allowed_predicates:
                        self.allowed_predicates[pred]['type'] = types.strip()
                elif line.startswith('direction('):
                    content = line[line.index('(')+1:line.rindex(')')]
                    pred, dirs = content.split(',', 1)
                    if pred in self.allowed_predicates:
                        self.allowed_predicates[pred]['direction'] = dirs.strip()

    def load_background_knowledge(self, bk_file: str):
        """Load background knowledge from file"""
        if not os.path.exists(bk_file):
            raise FileNotFoundError(f"Background knowledge file not found: {bk_file}")
            
        # Clear existing facts
        for pred_set in self.background_facts.values():
            pred_set.clear()
            
        with open(bk_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%') or line.startswith(':-'):
                    continue
                    
                if '(' in line and ').' in line:
                    pred = line[:line.index('(')]
                    args_str = line[line.index('(')+1:line.index(')')]
                    args = [arg.strip() for arg in args_str.split(',')]
                    
                    if len(args) == 2:
                        if pred == 'target':
                            protein, drug = args
                            self.known_proteins.add(protein)
                            self.known_drugs.add(drug)
                        elif pred.startswith('target'):
                            drug, protein = args
                            self.known_proteins.add(protein)
                            self.known_drugs.add(drug)
                        
                        if pred not in self.background_facts:
                            self.background_facts[pred] = set()
                        self.background_facts[pred].add((args[0], args[1]))

    def load_examples(self, examples_file: str):
        """Load examples from file and extract known drug names"""
        if not os.path.exists(examples_file):
            raise FileNotFoundError(f"Examples file not found: {examples_file}")
            
        self.positive_interactions.clear()
        self.negative_interactions.clear()
        
        with open(examples_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                match = re.match(r'(pos|neg)\(interacts\(([^,]+),([^)]+)\)\)', line)
                if match:
                    interaction_type, drug1, drug2 = match.groups()
                    self.known_drugs.add(drug1)
                    self.known_drugs.add(drug2)
                    if interaction_type == 'pos':
                        self.positive_interactions.add((drug1, drug2))
                    else:
                        self.negative_interactions.add((drug1, drug2))

    def add_fact(self, predicate: str, arg1: str, arg2: str) -> bool:
        """Add a fact to the knowledge base"""
        if predicate not in self.background_facts:
            self.background_facts[predicate] = set()
        
        self.background_facts[predicate].add((arg1, arg2))
        return True

    def add_interaction(self, drug1: str, drug2: str, positive: bool = True):
        """Add a drug-drug interaction example"""
        if positive:
            self.positive_interactions.add((drug1, drug2))
        else:
            self.negative_interactions.add((drug1, drug2))

    def _generate_bk_file(self, path: str):
        """Generate background knowledge file for Popper"""
        bk_content = []
        bk_content.append(":-style_check(-discontiguous).\n")
        
        for pred, fact_set in self.background_facts.items():
            for arg1, arg2 in fact_set:
                bk_content.append(f"{pred}({arg1},{arg2}).")
        
        with open(path, 'w') as f:
            f.write('\n'.join(bk_content))

    def _generate_examples_file(self, path: str):
        """Generate examples file for Popper"""
        examples = []
        
        for drug1, drug2 in self.positive_interactions:
            examples.append(f"pos(interacts({drug1},{drug2})).")
            
        for drug1, drug2 in self.negative_interactions:
            examples.append(f"neg(interacts({drug1},{drug2})).")
            
        with open(path, 'w') as f:
            f.write('\n'.join(examples))


    def find_rules(self, 
                   max_vars: int = 6,
                   timeout: int = 600) -> List[str]:
        """
        Use Popper to find rules from the current knowledge base
        
        Args:
            max_vars: Maximum number of variables in found rules
            timeout: Timeout in seconds
            
        Returns:
            List of found rules as strings
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy bias file instead of generating it
            tmp_bias_file = os.path.join(tmp_dir, 'bias.pl')
            shutil.copy(self.bias_file, tmp_bias_file)
            
            # Generate other necessary files
            bk_file = os.path.join(tmp_dir, 'bk.pl')
            examples_file = os.path.join(tmp_dir, 'exs.pl')
            
            self._generate_bk_file(bk_file)
            self._generate_examples_file(examples_file)
            
            # Read files into solver's internal state
            with open(tmp_bias_file, 'r') as f:
                self.bias_code = f.read()
            with open(bk_file, 'r') as f:
                self.bk_code = f.read()
            with open(examples_file, 'r') as f:
                self.examples_code = f.read()
            
            # Run Popper with retry loop for syntax errors
            for attempt in range(self.max_attempts):
                try:
                    log.info("Running Popper (attempt %d/%d)", attempt + 1, self.max_attempts)
                    success, error_msg, rules = self._run_solver()
                    if success and rules:
                        self.found_rules = rules
                        log.info("Found %d rules", len(self.found_rules))
                        return self.found_rules
                    break  # If we get here without solutions, no need to retry
                except Exception as e:
                    log.error("Error running Popper (attempt %d/%d): %s", 
                             attempt + 1, self.max_attempts, str(e))
                    if attempt < self.max_attempts - 1:  # Don't try to fix on last attempt
                        try:
                            fixed_code = self._fix_syntax_error({
                                'bias': self.bias_code,
                                'background': self.bk_code,
                                'examples': self.examples_code
                            }, str(e))
                            if fixed_code:
                                self.bias_code = fixed_code['bias']
                                self.bk_code = fixed_code['background']
                                self.examples_code = fixed_code['examples']
                                continue
                        except Exception as fix_error:
                            log.error("Failed to fix syntax error: %s", str(fix_error))
                    break  # Break if we can't fix or it's the last attempt
            
            log.warning("No rules found after %d attempts", attempt + 1)
            return []

    def predict_interaction(self, drug1: str, drug2: str) -> Optional[bool]:
        """Predict if two drugs interact using found rules"""
        log.info("Predicting interaction between %s and %s", drug1, drug2)
        
        if not self.found_rules:
            log.warning("No rules available for prediction. Run find_rules() first.")
            return None
            
        if drug1 not in self.known_drugs:
            log.error("Unknown drug: %s", drug1)
            return None
        if drug2 not in self.known_drugs:
            log.error("Unknown drug: %s", drug2)
            return None
            
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
                f.write(":-style_check(-discontiguous).\n\n")
                
                for pred, fact_set in self.background_facts.items():
                    for arg1, arg2 in fact_set:
                        f.write(f"{pred}({arg1},{arg2}).\n")
                f.write("\n")
                
                for rule in self.found_rules:
                    f.write(f"{rule}\n")
                f.write("\n")
                
                prolog_file = f.name
                
            prolog = Prolog()
            list(prolog.query("set_prolog_flag(verbose, silent), set_prolog_flag(verbose_load, false), style_check(-singleton)"))            
            prolog.consult(prolog_file)
            
            query = f"interacts({drug1},{drug2})"
            log.debug("Executing Prolog query: %s", query)
            results = list(prolog.query(query))
            
            os.unlink(prolog_file)
            
            interaction = len(results) > 0
            log.info("Prediction result: %s", "interaction found" if interaction else "no interaction")
            return interaction
            
        except ImportError:
            log.error("Error: pyswip not installed. Install with 'pip install pyswip'")
            return None
        except Exception as e:
            log.error("Error during Prolog evaluation: %s", str(e))
            return None