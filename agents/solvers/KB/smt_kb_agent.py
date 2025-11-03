# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from typing import List, Dict, Tuple, Optional, Any
import json
import re
import logging
from agents.base import BaseAgent, Scratchpad

log = logging.getLogger("agentic_sdk")

class SMTKnowledgeBaseAgent(BaseAgent):
    """Agent for managing SMT knowledge base operations (no solving)."""
    
    def __init__(self, name: str, goal: str):
        super().__init__(name, goal)
        
        # KB-specific prompts
        self.prompt_dict = {
            'get_rule_smt_form': 'get_rule_smt_form.txt',
            'get_instance_smt_form': 'get_instance_smt_form.txt',
            'refine_smt_error': 'refine_smt_error.txt'
        }
        
        log.info("Initialized SMTKnowledgeBaseAgent with prompts: %s", self.prompt_dict)

    def __call__(self, data: Any, memory: Scratchpad) -> Dict[str, Any]:
        """Main entry point for the KB agent."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        operation = data.get('operation')
        operations = {
            'add_rule': lambda: self.add_rule(memory.read("current_rule"), memory),
            'add_instance': lambda: self.add_instance(memory.read("current_instance"), memory),
            'get_rules': lambda: self._get_rules(memory),
            'get_instances': lambda: self._get_instances(memory),
            'get_parameters': lambda: self._get_parameters(memory),
            'create_problems_for_instance': lambda: self.create_problems_for_instance(memory.read("current_instance"), memory)
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        return operations[operation]()

    def _get_data(self, key: str, result_key: str, memory: Scratchpad) -> Dict[str, Any]:
        """Generic method to get data from memory."""
        data = memory.read(key) or []
        return {"success": True, result_key: data}

    def _get_rules(self, memory: Scratchpad) -> Dict[str, Any]:
        """Get all rules from the knowledge base."""
        return self._get_data("smt_rules", "rules", memory)

    def _get_instances(self, memory: Scratchpad) -> Dict[str, Any]:
        """Get all instances from the knowledge base."""
        return self._get_data("smt_instances", "instances", memory)

    def _get_parameters(self, memory: Scratchpad) -> Dict[str, Any]:
        """Get all parameters from the knowledge base."""
        return self._get_data("smt_all_parameters", "parameters", memory)

    def _extract_code_from_response(self, response: str) -> str:
        """Extract SMT-LIB code from LLM response."""
        code = response
        code_blocks = response.split("'''")
        if len(code_blocks) >= 3:
            code = code_blocks[1].strip()
        
        keywords = {"smt-lib", "smt", "smtlib"}
        lines = code.splitlines()
        filtered_lines = [line for line in lines if line.strip().lower() not in keywords]
        return "\n".join(filtered_lines).strip()

    def _extract_parameters_from_smt(self, smt_code: str) -> List[Dict[str, str]]:
        """Extract parameters and their types from SMT-LIB code."""
        pattern = r'\(declare-const\s+(\w+)\s+(Bool|Int|Real|String)\s*\)'
        matches = re.findall(pattern, smt_code)
        return [{"name": name, "type": typ} for name, typ in matches]

    def _get_smt_code(self, prompt_name: str, **kwargs) -> str:
        """Generic method to get SMT code from LLM."""
        all_parameters = kwargs.get('memory').read("smt_all_parameters") or []
        response = self.llm.generate(
            model_prompt_dir='kb',
            prompt_name=self.prompt_dict[prompt_name],
            parameters=json.dumps(all_parameters, indent=2),
            **{k: v for k, v in kwargs.items() if k != 'memory'}
        )
        return self._extract_code_from_response(response)

    def _update_parameters(self, new_parameters: List[Dict[str, str]], memory: Scratchpad) -> None:
        """Update global parameter list with new parameters."""
        all_parameters = memory.read("smt_all_parameters") or []
        for param in new_parameters:
            if not any(p["name"] == param["name"] for p in all_parameters):
                all_parameters.append(param)
            else:
                existing = next(p for p in all_parameters if p["name"] == param["name"])
                if existing["type"] != param["type"]:
                    raise ValueError(f"Type mismatch for parameter {param['name']}: previously {existing['type']}, now {param['type']}")
        memory.write("smt_all_parameters", all_parameters)

    def add_rule(self, natural_language_rule: str, memory: Scratchpad) -> Dict[str, Any]:
        """Add a new rule to the knowledge base."""
        try:
            smt_code = self._get_smt_code('get_rule_smt_form', rule=natural_language_rule, memory=memory)
            parameters = self._extract_parameters_from_smt(smt_code)
            
            rule = {
                "natural_language": natural_language_rule,
                "smt_code": smt_code,
                "parameters": parameters
            }
            
            rules = memory.read("smt_rules") or []
            rules.append(rule)
            memory.write("smt_rules", rules)
            
            self._update_parameters(parameters, memory)
            
            log.info("Rule added successfully")
            return {"success": True, "rule": rule}
        except Exception as e:
            log.error("Error processing rule: %s", str(e))
            return {"success": False, "error": str(e)}

    def _extract_parameter_assignments(self, smt_code: str) -> Tuple[List[str], List[str]]:
        """Extract parameter assignments from SMT-LIB code."""
        explicit_params, inferred_params = [], []
        current_section = None
        
        for line in smt_code.split('\n'):
            line = line.strip()
            if '; Explicitly specified parameters' in line:
                current_section = 'explicit'
            elif '; Inferred parameters' in line:
                current_section = 'inferred'
            elif line.startswith('(assert (='):
                param_name = line.split()[2]
                if current_section == 'explicit':
                    explicit_params.append(param_name)
                elif current_section == 'inferred':
                    inferred_params.append(param_name)
        
        return explicit_params, inferred_params

    def add_instance(self, instance_description: str, memory: Scratchpad) -> Dict[str, Any]:
        """Add a new patient instance to the knowledge base."""
        try:
            smt_code = self._get_smt_code('get_instance_smt_form', instance=instance_description, memory=memory)
            _, inferred_params = self._extract_parameter_assignments(smt_code)
            
            instance = {
                "description": instance_description,
                "smt_code": smt_code,
                "inferred_parameters": inferred_params
            }
            
            instances = memory.read("smt_instances") or []
            instances.append(instance)
            memory.write("smt_instances", instances)
            
            log.info("Instance added successfully")
            return {"success": True, "instance": instance}
        except Exception as e:
            log.error("Error processing instance: %s", str(e))
            return {"success": False, "error": str(e)}

    def create_problems_for_instance(self, instance_description: str, memory: Scratchpad) -> Dict[str, Any]:
        """Create SMT problems for an instance against all rules."""
        try:
            rules = memory.read("smt_rules") or []
            if not rules:
                return {"success": False, "error": "No rules found in knowledge base"}
            
            # Get or create instance
            instances = memory.read("smt_instances") or []
            existing_instance = next((inst for inst in instances 
                                   if inst.get("description", "").strip() == instance_description.strip()), None)
            
            if existing_instance:
                instance_smt = existing_instance["smt_code"]
                inferred_params = existing_instance["inferred_parameters"]
                log.info("Using existing formalized instance from memory")
                instance_stored = False
            else:
                smt_code = self._get_smt_code('get_instance_smt_form', instance=instance_description, memory=memory)
                _, inferred_params = self._extract_parameter_assignments(smt_code)
                instance_smt = smt_code
                
                new_instance = {
                    "description": instance_description,
                    "smt_code": instance_smt,
                    "inferred_parameters": inferred_params
                }
                instances.append(new_instance)
                memory.write("smt_instances", instances)
                log.info("Formalized and stored new instance in memory")
                instance_stored = True
            
            # Store current instance and create problem queue
            memory.write("smt_current_instance", {
                "description": instance_description,
                "smt_code": instance_smt,
                "inferred_parameters": inferred_params
            })
            
            problem_queue = [
                {
                    "rule_index": i,
                    "rule_natural_language": rule.get("natural_language", f"Rule {i}"),
                    "instance_natural_description": instance_description,
                    "smt_code": self.combine_rule_and_instance(rule, instance_smt),
                    "formalized": True
                }
                for i, rule in enumerate(rules)
            ]
            
            memory.write("smt_problem_queue", problem_queue)
            log.info("Created %d problems in queue for instance", len(problem_queue))
            
            return {
                "success": True,
                "problems_created": len(problem_queue),
                "instance_formalized": True,
                "instance_stored": instance_stored
            }
        except Exception as e:
            log.error("Error creating problems for instance: %s", str(e))
            return {"success": False, "error": str(e)}

    def combine_rule_and_instance(self, rule: dict, instance_code: str) -> str:
        """Combine rule and instance SMT code for evaluation."""
        rule_param_names = {p["name"] for p in rule.get("parameters", [])}
        filtered_instance_lines = []
        current_section = None
        section_params = {'explicit': [], 'inferred': []}
        
        for line in instance_code.split('\n'):
            line = line.strip()
            if not line:
                continue
            if '; Explicitly specified parameters' in line:
                current_section = 'explicit'
            elif '; Inferred parameters' in line:
                current_section = 'inferred'
            elif line.startswith('(assert'):
                param_match = re.search(r'\(assert\s+\(=\s+(\w+)\s+', line)
                if param_match and param_match.group(1) in rule_param_names and current_section:
                    section_params[current_section].append(line)
        
        if section_params['explicit']:
            filtered_instance_lines.extend(['; Explicitly specified parameters'] + section_params['explicit'])
        if section_params['inferred']:
            if filtered_instance_lines:
                filtered_instance_lines.append('')
            filtered_instance_lines.extend(['; Inferred parameters'] + section_params['inferred'])
        
        return f"""
            {rule['smt_code'].strip()}

            ; Instance assignments (filtered to rule parameters)
            {chr(10).join(filtered_instance_lines).strip()}

            (check-sat)
            """

    def save(self, filepath: str, memory: Scratchpad) -> Dict[str, Any]:
        """Save the knowledge base to a file."""
        try:
            data = {
                'rules': memory.read("smt_rules") or [],
                'instances': memory.read("smt_instances") or [],
                'parameters': memory.read("smt_all_parameters") or []
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            log.info("Knowledge base saved to %s", filepath)
            return {"success": True}
        except Exception as e:
            log.error("Error saving knowledge base: %s", str(e))
            return {"success": False, "error": str(e)}
        
    def load(self, filepath: str, memory: Scratchpad) -> Dict[str, Any]:
        """Load a knowledge base from a file."""
        try:
            log.info("Loading knowledge base from %s", filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            memory.write("smt_all_parameters", data.get('parameters', []))
            memory.write("smt_rules", data.get('rules', []))
            memory.write("smt_instances", data.get('instances', []))
            
            log.info("Loaded %d rules, %d instances, and %d parameters", 
                    len(data.get('rules', [])), len(data.get('instances', [])), len(data.get('parameters', [])))
            
            return {"success": True}
        except Exception as e:
            log.error("Error loading knowledge base: %s", str(e))
            return {"success": False, "error": str(e)}


