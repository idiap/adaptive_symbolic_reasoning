# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from abc import ABC, abstractmethod
import re
import json
import copy
from typing import Optional, Dict

from base.logger import setup_logging
log = setup_logging()


class GenerativeModel(ABC):
    def __init__(self, model_name,
                 prompt_model=None):
        self.model_name = model_name
        self.prompt_model = prompt_model

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass
    
    def _safe_json(self, text: str) -> Dict:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        self.json_pat = re.compile(r"\{.*\}", re.S)
        m = self.json_pat.search(text)
        if not m:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        try:
            return json.loads(m.group())
        except:
            return json.loads('['+ m.group()+']')

    def extract_code(self, result):
        result = result.replace('```json', '```')
        pattern = r"```(.*?)```"
        match = re.search(pattern, result, re.DOTALL)
        if match:
            result = match.group(1)
        return result
    
    def set_model_args(self, model_args: Dict):
        old_param_backup = {}
        
        # First, backup model_kwargs if it exists and we have parameters to set
        if hasattr(self.client, 'model_kwargs') and model_args:
            current_model_kwargs = getattr(self.client, 'model_kwargs')
            old_param_backup['model_kwargs'] = copy.deepcopy(current_model_kwargs) if isinstance(current_model_kwargs, dict) else {}
        
        for key, value in model_args.items():
            if hasattr(self.client, key):
                if key not in old_param_backup:  # Only backup if not already backed up
                    old_param_backup[key] = getattr(self.client, key)
                setattr(self.client, key, value)
            else:
                # Handle model_kwargs specifically
                if hasattr(self.client, 'model_kwargs'):
                    # Get current model_kwargs and add the new parameter
                    current_model_kwargs = getattr(self.client, 'model_kwargs')
                    new_model_kwargs = copy.deepcopy(current_model_kwargs) if isinstance(current_model_kwargs, dict) else {}
                    new_model_kwargs[key] = value
                    setattr(self.client, 'model_kwargs', new_model_kwargs)
                else:
                    log.warning(f"Model config does not have attribute 'model_kwargs'")
        
        return old_param_backup

    def extract_numbered_list(self, model_reponse: str,
                          prefix: Optional[str] = None,
                          remove_number: Optional[bool] = False) -> str:
        """
        Extract content after a specific prefix or numbered list.
        
        Parameters:
        - model_reponse (str): Input string from which to extract content
        - prefix (str, optional): Prefix to match. If provided, extract content after this prefix
        - remove_number (bool, optional): If True, remove numbers from the extracted content
        """
        if prefix:
            parts = model_reponse.split(prefix, 1)
            if len(parts) > 1:
                extracted = parts[1].strip()
                
                if remove_number and re.search(r'^\d+\.', extracted, re.MULTILINE):
                    cleaned = re.sub(r'\d+\.\s*', '', extracted)
                    return '\n'.join(cleaned.splitlines()).strip()
                else:
                    return extracted
            else:
                extracted = model_reponse
        else:
            numbered_list_pattern = re.compile(r'(\d+\..*?(?:\n|$))+',
                                            re.IGNORECASE | re.DOTALL)
            match = numbered_list_pattern.search(model_reponse)
            extracted = match.group(0) if match else model_reponse

        if remove_number:
            cleaned = re.sub(r'\d+\.\s*', '', extracted)
            return '\n'.join(cleaned.splitlines()).strip()
        else:
            return extracted