#!/bin/bash

data_name="trec" # testing data

# test on api models
# backbone_type="gpt-4o-openai"
backbone_type="gpt-4o-azure"

# # or test on local model
# backbone_type="qwen2.5-coder-7b"
# backbone_type="qwen2.5-7b"
# backbone_type="llama-3.1-8b"

## test ours
python _tester.py --data_name ${data_name} --backbone_type ${backbone_type}
