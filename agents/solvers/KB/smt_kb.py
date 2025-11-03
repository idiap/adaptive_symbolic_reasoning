# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from typing import List, Dict, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import json
import re
from agents.solvers.smt_solver import SMTSolver
import logging

log = logging.getLogger("agentic_sdk")

class SMTKnowledgeBase(SMTSolver):
    def __init__(self, name: str, goal: str, initial_rules: Optional[List[str]] = None, load_from: Optional[str] = None):
        """
        Initialize the SMT Knowledge Base
        trial_descriptions
        Args:
            name: Name of the agent
            goal: Goal of the agent
            initial_rules: Optional list of natural language rules to initialize the KB with
            load_from: Optional path to load a previously saved knowledge base
        """
        # Initialize parent class first to get its prompt dictionary
        super().__init__(name, goal)
        
        # Add KB-specific prompts to the existing prompt dictionary
        kb_prompts = {
            'get_rule_smt_form': 'get_rule_smt_form.txt',
            'get_instance_smt_form': 'get_instance_smt_form.txt',
            'refine_smt_error': 'refine_smt_error.txt'
        }
        
        # Append KB prompts to parent's prompt dictionary
        self.prompt_dict.update(kb_prompts)
        
        self.rules: List[Dict[str, Any]] = []
        self.all_parameters: List[Dict[str, str]] = []  # List of all unique parameters used across rules
        
        log.info("Initialized SMTKnowledgeBase with prompts: %s", self.prompt_dict)
        
        if load_from:
            self.load(load_from)
        # Process initial rules if provided and not loading from file
        elif initial_rules:
            log.info("Processing initial rules...")
            for rule in initial_rules:
                self.add_rule(rule)

    def _extract_parameters_from_smt(self, smt_code: str) -> List[Dict[str, str]]:
        """
        Extract parameters and their types from SMT-LIB code by parsing declare-const statements
        
        Args:
            smt_code: The SMT-LIB code to parse
            
        Returns:
            List of Parameters with names and types
        """
        # Match (declare-const param_name Type) pattern for all supported types
        pattern = r'\(declare-const\s+(\w+)\s+(Bool|Int|Real)\)'
        matches = re.findall(pattern, smt_code)
        return [{"name": name, "type": type} for name, type in matches]

    def _get_smt_rule_code(self, natural_language_rule: str) -> str:
        """
        Use LLM to generate SMT code for a rule
        
        Args:
            natural_language_rule: The rule in natural language
            
        Returns:
            SMT-LIB code string
        """
        # Convert parameters to dict format for JSON serialization
        param_dicts = self.all_parameters
        
        # Call LLM with proper prompt structure
        response = self.llm.generate(
            model_prompt_dir='kb',
            prompt_name=self.prompt_dict['get_rule_smt_form'],
            parameters=json.dumps(param_dicts, indent=2),
            rule=natural_language_rule
        )
        
        return self._extract_code_from_response(response)
    
    def add_rule(self, natural_language_rule: str) -> None:
        """
        Add a new eligibility rule to the knowledge base
        
        Args:
            natural_language_rule: The rule in natural language
        """
        
        try:
            # Get SMT code from LLM
            smt_code = self._get_smt_rule_code(natural_language_rule)            
            # Extract parameters from SMT code
            parameters = self._extract_parameters_from_smt(smt_code)
            
            # Update all_parameters with any new parameters
            for param in parameters:
                if not any(p["name"] == param["name"] for p in self.all_parameters):
                    self.all_parameters.append(param)
                else:
                    # Verify type consistency
                    existing = next(p for p in self.all_parameters if p["name"] == param["name"])
                    if existing["type"] != param["type"]:
                        raise ValueError(f"Type mismatch for parameter {param['name']}: previously {existing['type']}, now {param['type']}")
            
            # Create and store new rule
            rule = {
                "natural_language": natural_language_rule,
                "smt_code": smt_code,
                "parameters": parameters
            }
            self.rules.append(rule)
            log.info("Rule added.")
        except Exception as e:
            log.error("Error processing rule: %s", str(e))
            raise ValueError(f"Failed to process eligibility rule: {e}")

    def get_all_parameters(self) -> List[Dict[str, str]]:
        """
        Get list of all unique parameters with their types
        """
        return self.all_parameters
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Get list of rules with their parameters
        
        Returns:
            List of dictionaries containing rule information
        """
        return [{
            "natural_language": rule["natural_language"],
            "smt_code": rule["smt_code"],
            "parameters": rule["parameters"]
        } for rule in self.rules]

    def _extract_parameter_assignments(self, smt_code: str) -> Tuple[List[str], List[str]]:
        """
        Extract parameter assignments from SMT-LIB code, separating explicit and inferred.
        
        Args:
            smt_code: SMT-LIB code containing parameter assignments
            
        Returns:
            Tuple of (explicit parameter names, inferred parameter names)
        """
        explicit_params = []
        inferred_params = []
        current_section = None
        
        for line in smt_code.split('\n'):
            line = line.strip()
            
            if '; Explicitly specified parameters' in line:
                current_section = 'explicit'
                continue
            elif '; Inferred parameters' in line:
                current_section = 'inferred'
                continue
                
            if line.startswith('(assert (='):
                # Extract parameter name from (assert (= param_name value))
                param_name = line.split()[2]
                if current_section == 'explicit':
                    explicit_params.append(param_name)
                elif current_section == 'inferred':
                    inferred_params.append(param_name)
                    
        return explicit_params, inferred_params

    def _formalize_instance(self, instance_description: str) -> Tuple[str, List[str]]:
        """
        Convert instance description to SMT-LIB parameter assignments.
        
        Args:
            instance_description: Natural language description of the instance
            
        Returns:
            Tuple of (SMT-LIB code, list of inferred parameter names)
        """
        param_dicts = self.all_parameters
        
        response = self.llm.generate(
            model_prompt_dir='kb',
            prompt_name=self.prompt_dict['get_instance_smt_form'],
            parameters=json.dumps(param_dicts, indent=2),
            instance=instance_description
        )
        
        try:
            smt_code = self._extract_code_from_response(response)
            _, inferred_params = self._extract_parameter_assignments(smt_code)
            return smt_code, inferred_params
        except ValueError as e:
            log.error("Failed to extract SMT code. Response analysis:")
            log.error("Response length: %d", len(response))
            log.error("First 500 characters: %s", response[:500])
            log.error("Does response contain markers:")
            triple_quotes = "'''"
            log.error("- Triple quotes: %d", response.count(triple_quotes))
            log.error("- SMT-LIB keyword: %s", 'smt-lib' in response.lower())
            log.error("- declare-const: %s", 'declare-const' in response)
            log.error("- assert: %s", 'assert' in response)
            raise

    def _evaluate_instance_for_one_rule(self, rule: Dict[str, Any], instance_smt: str) -> Tuple[bool, bool, str, str]:
        """
        Evaluate a rule against instance assignments with refinement loop.
        
        Args:
            rule: The rule to evaluate
            instance_smt: SMT-LIB code for instance parameter assignments
            
        Returns:
            Tuple of (success, is_satisfied, error_message, combined_smt_code)
        """
        # Get set of parameters used in this rule
        rule_param_names = {p["name"] for p in rule["parameters"]}
        
        # Filter instance assignments to only include relevant parameters
        filtered_instance_lines = []
        current_section = None
        section_params = {'explicit': [], 'inferred': []}
        
        # First pass: collect parameters for each section
        for line in instance_smt.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if '; Explicitly specified parameters' in line:
                current_section = 'explicit'
                continue
            elif '; Inferred parameters' in line:
                current_section = 'inferred'
                continue
            
            if line.startswith('(assert'):
                param_match = re.search(r'\(assert\s+\(=\s+(\w+)\s+', line)
                if param_match and param_match.group(1) in rule_param_names:
                    if current_section:
                        section_params[current_section].append(line)
        
        # Second pass: build output with sections that have parameters
        if section_params['explicit']:
            filtered_instance_lines.append('; Explicitly specified parameters')
            filtered_instance_lines.extend(section_params['explicit'])
            
        if section_params['inferred']:
            if filtered_instance_lines:  # Add blank line between sections
                filtered_instance_lines.append('')
            filtered_instance_lines.append('; Inferred parameters')
            filtered_instance_lines.extend(section_params['inferred'])
        
        filtered_instance_smt = '\n'.join(filtered_instance_lines)
        
        current_code = f"""
        ; Rule (includes parameter declarations)
        {rule["smt_code"]}
        
        ; Instance assignments
        {filtered_instance_smt}
        
        (check-sat)
        """
        
        for attempt in range(self.max_attempts):
            try:
                # Run Z3 using the inherited _run_solver method
                success, error_msg, solutions = self._run_solver(current_code)
                if success:
                    return True, bool(solutions), "", current_code
                
                if attempt == self.max_attempts - 1:
                    return False, False, f"Failed to evaluate after {self.max_attempts} attempts. Last error: {error_msg}", current_code
                
                log.warning("Attempt %d failed. Error: %s", attempt + 1, error_msg)
                log.info("Attempting refinement...")
                
                # Try to refine the code using LLM
                try:
                    current_code = self._fix_syntax_error(current_code, error_msg)
                    log.info("Refinement succeeded on attempt %d.", attempt + 1)
                    log.debug("Refined code: %s", current_code)
                except Exception as e:
                    log.error("Refinement failed: %s", str(e))
                    if attempt == self.max_attempts - 1:
                        return False, False, f"Refinement failed: {str(e)}", current_code
                
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    return False, False, str(e), current_code
                log.warning("Error in attempt %d: %s", attempt + 1, str(e))
                log.info("Retrying...")
        
        return False, False, "Failed to evaluate after all attempts", current_code

    def evaluate_instance(self, instance_description: str, verbose: bool = False, rule_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate an instance against all or a specific rule in the knowledge base.
        
        Args:
            instance_description: Natural language description of the instance
            verbose: If True, print progress as rules are evaluated
            rule_index: If set, only evaluate the rule at this index (supports negative indices)
            
        Returns:
            List of dictionaries containing rule evaluation results
        """
        try:
            # Formalize the instance
            if verbose:
                log.info("Formalizing instance description...")
            instance_smt, inferred_params = self._formalize_instance(instance_description)
            results = []

            log.debug("instance_smt: %s", instance_smt)
            
            if rule_index is not None:
                rules_to_eval = [self.rules[rule_index]]
            else:
                rules_to_eval = self.rules
            total_rules = len(rules_to_eval)
            for i, rule in enumerate(rules_to_eval, 1):
                if verbose:
                    log.info("Evaluating rule %d/%d...", i, total_rules)
                    log.info("Rule: %s...", rule["natural_language"][:200])
                
                try:
                    # Get rule param names for filtering inferred parameters
                    rule_param_names = {p["name"] for p in rule["parameters"]}
                    
                    # Extract values for inferred parameters from SMT code
                    inferred_with_values = []
                    for param_name in inferred_params:
                        if param_name in rule_param_names:
                            param_type = next(p["type"] for p in self.all_parameters if p["name"] == param_name)
                            # Look for the assertion in the SMT code that sets this parameter
                            match = re.search(rf'\(assert\s+\(=\s+{param_name}\s+([^)]+)\)', instance_smt)
                            if match:
                                value = match.group(1)
                                inferred_with_values.append({
                                    "name": param_name,
                                    "type": param_type,
                                    "value": value
                                })
                    
                    # Evaluate with refinement
                    success, is_sat, error_msg, combined_code = self._evaluate_instance_for_one_rule(rule, instance_smt)
                    
                    result = {
                        "is_satisfied": is_sat if success else False,
                        "rule_natural_language": rule["natural_language"],
                        "inferred_parameters": inferred_with_values,
                        "error": error_msg if not success else None,
                        "smt_code": combined_code
                    }
                    
                    if verbose:
                        log.info("Satisfied: %s", result["is_satisfied"])
                        if result["error"]:
                            log.warning("Error: %s", result["error"])
                        if result["inferred_parameters"]:
                            log.info("PEIRCE inferred the following parameters that were not specified:")
                            for param in result["inferred_parameters"]:
                                log.info("- %s (%s) = %s", param['name'], param['type'], param['value'])
                    
                    results.append(result)
                        
                except Exception as e:
                    if verbose:
                        log.error("Error evaluating rule: %s", str(e))
                    
                    results.append({
                        "is_satisfied": False,
                        "rule_natural_language": rule["natural_language"],
                        "inferred_parameters": [],
                        "error": str(e),
                        "smt_code": None
                    })
            
            return results
            
        except Exception as e:
            raise ValueError(f"Failed to evaluate instance: {str(e)}")

    def save(self, filepath: str):
        """
        Save the knowledge base to a file.
        
        Args:
            filepath: Path to save the knowledge base to
        """
        # Convert rules and parameters to serializable format
        data = {
            'rules': [
                {
                    'natural_language': rule['natural_language'],
                    'smt_code': rule['smt_code'],
                    'parameters': rule['parameters']
                }
                for rule in self.rules
            ],
            'parameters': self.all_parameters
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        log.info("Knowledge base saved to %s", filepath)
        
    def load(self, filepath: str):
        """
        Load a knowledge base from a file.
        
        Args:
            filepath: Path to load the knowledge base from
        """
        log.info("Loading knowledge base from %s", filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load parameters
        self.all_parameters = [
            {"name": p['name'], "type": p['type']}
            for p in data['parameters']
        ]
        
        # Load rules
        self.rules = [
            {
                "natural_language": r['natural_language'],
                "smt_code": r['smt_code'],
                "parameters": r['parameters']
            }
            for r in data['rules']
        ]
        
        log.info("Loaded %d rules and %d parameters", len(self.rules), len(self.all_parameters))
