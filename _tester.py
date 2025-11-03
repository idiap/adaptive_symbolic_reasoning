# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import os

import warnings
warnings.filterwarnings("ignore")

from agents.generation.api import OpenAIGenerator, AzureOpenAIGenerator, OpenAICompatibleGenerator, GeminiGenerator
from agents.generation.local import LocalGenerator
import yaml
import argparse

from comparing_methods.llm_planned import evaluating_llm_planned
from comparing_methods.llm_vanilla import evaluating_llm
from comparing_methods.carnap import evaluating_agent
from comparing_methods.llm_symbolCoT import evaluating_llm_symbolCoT
from comparing_methods.llm_symbolCoT_planned import evaluating_llm_symbolCoT_planned
from base.logger import setup_logging
log = setup_logging()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="folio")
    parser.add_argument("--max_testing_num", type=int, default=-1)
    parser.add_argument("--test_setting", type=str, default="agent")
    parser.add_argument("--use_cot", action="store_true") 
    parser.add_argument("--backbone_type", type=str, default="gpt-4o-azure")

    args = parser.parse_args()

    test_data_name = args.data_name
    max_examples = args.max_testing_num
    test_setting = args.test_setting
    use_cot = args.use_cot

    # load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if args.backbone_type == 'gpt-4o-azure':
        # Azure OpenAI
        api_key = config.get('api_config', {}).get('gpt-4o-azure', {}).get('api_key')
        model_name = config.get('api_config', {}).get('gpt-4o-azure', {}).get('model_name')
        model_version = config.get('api_config', {}).get('gpt-4o-azure', {}).get('openai_api_version')
        azure_endpoint = config.get('api_config', {}).get('gpt-4o-azure', {}).get('azure_endpoint')
        llm = AzureOpenAIGenerator(model_name=model_name, api_key=api_key, model_version=model_version, azure_endpoint=azure_endpoint)
    elif args.backbone_type == 'gpt-4o-openai':
        # OpenAI Official
        api_key = config.get('api_config', {}).get('gpt-4o-openai', {}).get('api_key')
        model_name = config.get('api_config', {}).get('gpt-4o-openai', {}).get('model_name')
        llm = OpenAIGenerator(model_name=model_name, api_key=api_key)
    elif args.backbone_type == 'gemini':
        # Google Gemini
        api_key = config.get('api_config', {}).get('gemini', {}).get('api_key')
        llm = GeminiGenerator(model_name='gemini', api_key=api_key)
    elif args.backbone_type == 'deepseek-v3':
        # OpenAI Compatible API (DeepSeek)
        api_key = config.get('api_config', {}).get('deepseek-v3', {}).get('api_key')
        model_name = config.get('api_config', {}).get('deepseek-v3', {}).get('model_name')
        base_url = config.get('api_config', {}).get('deepseek-v3', {}).get('base_url')
        llm = OpenAICompatibleGenerator(model_name=model_name, api_key=api_key, base_url=base_url)
    else:
        # Local models
        api_key = config.get('api_config', {}).get(args.backbone_type, {}).get('api_key')
        model_name = config.get('api_config', {}).get(args.backbone_type, {}).get('model_name')
        lora_path = config.get('api_config', {}).get(args.backbone_type, {}).get('lora_path', None)
        llm = LocalGenerator(model_name=model_name, api_key=api_key, lora_path=lora_path)

    log.info('='*50)
    log.info(f"Using {args.backbone_type} model for generation.")
    if max_examples == -1:
        log_str = f"Testing {test_setting} model on {test_data_name} dataset."
    else:
        log_str = f"Testing {test_setting} model on {max_examples} examples of {test_data_name} dataset."
    if use_cot and test_setting == 'agent':
        log_str += " Using CoT reasoning."
    log.info(log_str)
    log.info('='*50)

    args.backbone_type = args.backbone_type.replace('_openai', '').replace('_azure', '')
    
    evaluating_agent(test_data_name, llm, max_examples=max_examples, backbone_type = args.backbone_type)
