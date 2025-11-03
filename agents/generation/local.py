# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import os
import copy
import torch
import inspect
import json
from prompt.prompt_model import PromptModel
from typing import Optional, Dict, Any
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import HumanMessage, SystemMessage
from .abstract import GenerativeModel
from base.logger import setup_logging
from peft import PeftModel


log = setup_logging()

class LocalGenerator(GenerativeModel):
    def __init__(self, model_name = 'Qwen/Qwen2.5-1.5B-Instruct', prompt_model=None, api_key = None, lora_path=None):
        """
        Initialize LocalGenerator with optional LoRA adapter support.

        Args:
            model_name (str): Base model name
            prompt_model: Prompt model instance
            api_key (str): API key (if needed)
            lora_path (str, optional): LoRA adapter path, if provided will load and merge LoRA parameters
        """
        super().__init__(model_name, prompt_model)
        log.info(f'Using local model: {model_name}')
        if prompt_model is None:
            self.prompt_model = PromptModel()
        
        # Store LoRA path for later use
        self.lora_path = lora_path
        
        # Initialize generation parameters
        self.generation_params = {
            'max_new_tokens': 4096,
            'temperature': 0.01,
            'do_sample': True,
        }
        
        # Load model and tokenizer
        log.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token = api_key
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto',
            token = api_key
        )
        
        # Load and merge LoRA adapter if provided
        if self.lora_path:
            log.info(f"Loading LoRA adapter from: {self.lora_path}")
            try:
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                log.info("LoRA adapter loaded successfully")
                
                # Merge LoRA weights with base model for better inference performance
                log.info("Merging LoRA weights with base model...")
                self.model = self.model.merge_and_unload()
                log.info("LoRA weights merged successfully")
                
            except Exception as e:
                log.error(f"Failed to load LoRA adapter: {e}")
                log.warning("Continuing with base model only")
        
        log.info(f"Model loaded on device: {self.model.device}")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get supported parameters once during initialization
        self.all_supported_params = self._get_supported_params()
    
    def is_lora_enabled(self) -> bool:
        """Check if LoRA adapter is enabled"""
        return self.lora_path is not None
    
    def _get_supported_params(self):
        """Get all supported parameters for model generation"""
        supported_params = set()
        
        # Get from generate method signature
        try:
            generate_sig = inspect.signature(self.model.generate)
            supported_params.update(generate_sig.parameters.keys())
        except Exception as e:
            pass
        
        # Get from generation_config
        if hasattr(self.model, 'generation_config') and self.model.generation_config:
            config_params = set(dir(self.model.generation_config))
            # Filter out private attributes and methods
            config_params = {p for p in config_params if not p.startswith('_')}
            supported_params.update(config_params)
        
        return supported_params
    
    def set_model_args(self, model_args: Dict):
        """
        Dynamic parameter validation using pre-computed supported parameters.
        """
        # Backup current parameters
        old_param_backup = copy.deepcopy(self.generation_params)
        
        # Update parameters
        for key, value in model_args.items():
            # Handle parameter mapping
            if key == 'max_tokens':
                # Map max_tokens to max_new_tokens
                if 'max_new_tokens' in self.all_supported_params:
                    self.generation_params['max_new_tokens'] = value
                else:
                    log.warning("max_new_tokens not supported by model")
            elif key in self.all_supported_params or key in ['temperature', 'do_sample', 'top_p', 'top_k', 'max_new_tokens']:
                # Direct parameter assignment
                self.generation_params[key] = value
            elif key in ['response_format']:
                # Skip non-generation parameters
                pass
            else:
                log.warning(f"Parameter {key} not supported by model generation")
        
        return old_param_backup
    
    def completion_with_backoff(self, messages):
        """
        Generate completion with OpenAI-like interface.
        No retry logic needed for local models.
        """
        try:
            # Convert langchain messages to dict format
            if hasattr(messages[0], 'content'):
                formatted_messages = [{"role": msg.type if msg.type != 'human' else 'user', "content": msg.content} for msg in messages]
            else:
                formatted_messages = messages
            
            # Use chat template for proper formatting
            text = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Use dynamic parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **self.generation_params,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the new tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Handle thinking tokens
            if '</think>' in response_text:
                response_text = response_text.split('</think>')[-1]
                
            # Return OpenAI-like response format
            return type('Response', (), {
                'content': response_text.strip()
            })()
            
        except Exception as e:
            log.error(f"Error in completion_with_backoff: {e}")
            return type('Response', (), {
                'content': ''
            })()
    
    def __call__(self, messages, **model_args):
        """
        Quick call to the LLM with a single message.
        Compatible with APIGenerator interface.
        """
        # Set model parameters
        old_param_backup = self.set_model_args(model_args)
        
        try:
            # Call completion method
            response = self.completion_with_backoff(messages)
            result = response.content

            if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
                log.debug("JSON format requested, attempting enhanced parsing...")
                try:
                    result = self._enhanced_json_parsing(result)
                        
                except Exception as e:
                    log.error(f"Critical error in JSON parsing: {e}")
                    # Keep original result as fallback
                    result = result
            
            # Restore parameters
            _ = self.set_model_args(old_param_backup)
            
            return result
            
        except Exception as e:
            log.error(f"Error in __call__: {e}")
            # Restore parameters
            _ = self.set_model_args(old_param_backup)
            return ""

    def generate(self,
                 model_prompt_dir: str,
                 prompt_name: str,
                 prefix: Optional[str] = None,
                 numbered_list: Optional[bool] = False,
                 remove_number: Optional[bool] = False,
                 test: Optional[bool] = False,
                 model_args: Optional[dict] = {},
                 **replacements) -> str:
        """
        Generates a response from the Qwen model.
        Compatible with your existing generate interface.
        """
        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_name=model_prompt_dir,
            prompt_name=prompt_name,
            **replacements
        )
        # Adjust the model configuration
        old_param_backup = self.set_model_args(model_args)

        result = None
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **self.generation_params
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if '</think>' in result:
                result = result.split('</think>')[-1]

            if test:
                log.debug(result)

            if "```" in result:
                result = self.extract_code(result)

            if prefix and prefix in result:
                parts = result.split(prefix, 1)
                if len(parts) > 1:
                    result = parts[1].strip()

                    if numbered_list and remove_number:
                        result = re.sub(r'\d+\.\s*', '', result)
                        result = '\n'.join(result.splitlines()).strip()
            elif numbered_list:
                result = self.extract_numbered_list(result, None, remove_number)
            
            # Handle JSON format response with enhanced parsing
            if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
                log.debug("JSON format requested, attempting enhanced parsing...")
                try:
                    result = self._enhanced_json_parsing(result)
                        
                except Exception as e:
                    log.error(f"Critical error in JSON parsing: {e}")
                    # Keep original result as fallback
                    result = result
            
            # Restore the model configuration
            _ = self.set_model_args(old_param_backup)
            return result

        except Exception as e:
            log.debug(e)
            # Restore the model configuration
            _ = self.set_model_args(old_param_backup)
            return None

    def _format_messages(self, messages):
        # Mimic OpenAI chat format for Qwen
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _extract_assistant_response(self, response):
        # Extract only the assistant's reply
        if response is None:
            return ""
        return response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    def _enhanced_json_parsing(self, text: str) -> Dict[str, Any]:
        """
        Enhanced JSON parsing with multiple fallback strategies.
        Since open-source models don't have native JSON format support,
        we need robust extraction methods.
        """
        original_text = text
        log.debug(f"Attempting JSON parsing on text: {text[:200]}...")
        
        # Strategy 1: Direct JSON parsing
        try:
            # Try parsing the entire text as JSON
            cleaned_text = text.strip()
            if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                result = json.loads(cleaned_text)
                log.debug("Strategy 1 (direct parsing) succeeded")
                return result
        except json.JSONDecodeError:
            log.debug("Strategy 1 (direct parsing) failed")
        
        # Strategy 2: Extract JSON from code blocks
        try:
            # Look for JSON in code blocks
            patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'```json\s*(\[.*?\])\s*```',
                r'```\s*(\[.*?\])\s*```'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        result = json.loads(match.strip())
                        log.debug("Strategy 2 (code block extraction) succeeded")
                        return result
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.debug(f"Strategy 2 (code block extraction) failed: {e}")
        
        # Strategy 3: Find JSON-like structures in text
        try:
            # Look for balanced braces/brackets
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
                r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'  # Nested arrays
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        result = json.loads(match.strip())
                        log.debug("Strategy 3 (pattern matching) succeeded")
                        return result
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.debug(f"Strategy 3 (pattern matching) failed: {e}")
        
        # Strategy 4: Key-value extraction as fallback
        try:
            result = self._extract_key_values(text)
            if result:
                log.debug("Strategy 4 (key-value extraction) succeeded")
                return result
        except Exception as e:
            log.debug(f"Strategy 4 (key-value extraction) failed: {e}")
        
        # Strategy 5: Last resort - return structured error
        log.warning(f"All JSON parsing strategies failed for text: {original_text[:100]}...")
        return {
            "error": "JSON parsing failed",
            "original_text": original_text,
            "strategies_attempted": ["direct", "code_block", "pattern_matching", "key_value"]
        }
    
    def _extract_key_values(self, text: str) -> Dict[str, Any]:
        """
        Extract key-value pairs from free text when JSON parsing fails.
        This is a fallback method for when models don't produce valid JSON.
        """
        result = {}
        
        # Pattern 1: "key": "value" or "key": value
        kv_patterns = [
            r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
            r'"([^"]+)"\s*:\s*([^,\n\}]+)',  # "key": value
            r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"([^"]*)"',  # key: "value"
            r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,\n\}]+)',  # key: value
        ]
        
        for pattern in kv_patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                try:
                    # Try boolean
                    if value.lower() in ['true', 'false']:
                        result[key] = value.lower() == 'true'
                    # Try number
                    elif value.replace('.', '').replace('-', '').isdigit():
                        result[key] = float(value) if '.' in value else int(value)
                    # Try null
                    elif value.lower() == 'null':
                        result[key] = None
                    # Keep as string
                    else:
                        result[key] = value.strip('"\'')
                except:
                    result[key] = value.strip('"\'')
        
        # Pattern 2: Extract specific fields commonly used in reasoning tasks
        common_fields = ['answer', 'label', 'result', 'prediction', 'conclusion', 'reasoning']
        for field in common_fields:
            # Look for field: value patterns
            pattern = rf'{field}\s*[:\-=]\s*(.+?)(?:\n|$|,|\}})'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result[field] = matches[0].strip().strip('"\'')
        
        return result
    
    def _safe_json(self, text: str) -> Any:
        """
        Override the parent _safe_json method with enhanced parsing.
        This method is called when response_format is json_object.
        """
        return self._enhanced_json_parsing(text)