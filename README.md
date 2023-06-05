# EFactSum
This repository contains the code for the paper: Improving Factuality of Abstractive Summarization without Sacrificing Summary Quality **ACL 2023**

## Quick Links

- [Overview](#overview)
- [Generating data](#generating-data)
- [Training](#training)
- [Decoding](#decoding)
- [Model Outputs](#model-outputs)
- [Acknowlegements](#acknowlegements)

## Overview

## Generating data

## Training

## Decoding


## Model Outputs
We summarize the outputs from our models below

|          | Source | Model Output | Reference Output |
|----------|---------|---------|---------|
| CNNDM    | [cnndm.test.source](output/cnndm.test.source.txt) | BART - [cnndm.test.ours](output/cnndm.test.ours.txt) | [cnndm.test.target](output/cnndm.test.target.txt)  |
| XSum     | [xsum.test.source](output/xsum.test.source.txt) |  PEGASUS - [xsum.test.ours](output/xsum.test.ours.txt) | [xsum.test.target](output/xsum.test.target.text)  |



You can load our trained models from Huggingface Transformers.
Our model checkpoint on CNNDM (`tanay/efactsum-bart-cnndm`) is a standard BART model (i.e., `BartForConditionalGeneration`) while our model checkpoint on XSum (`tanay/efactsum-pegasus-xsum`) is a standard Pegasus model (i.e., `PegasusForConditionalGeneration`).

Example usage with HuggingFace: 

```python
from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration

IS_CNNDM = True
max_length = 1024 if IS_CNNDM else 512

ARTICLE_TO_SUMMARIZE = "firefighters responded to cries for help - from two parrots. the crew scoured a burning home in boise, idaho, searching \
for people shouting 'help!' and 'fire!' eventually, to their surprise, they found a pair of squawking birds. \
scroll down for video. cry for help! this is one of the two parrots who were found in a burning home after calling for help. \
the tropical creatures appeared to have been alone when flames began to sweep the property. but they seemed to know what to do. \
treatment: the officials treated the birds with oxygen masks and both are expected to survive. according to kboi, the cause of the officers \
managed to contain the fire to just one room. it is being investigated and no people were found inside. officials have yet to track down the birds' owners. .\ 
"

if IS_CNNDM:
    model = BartForConditionalGeneration.from_pretrained('tanay/efactsum-bart-cnndm')
    tokenizer = BartTokenizer.from_pretrained('tanay/efactsum-bart-cnndm')
else:
    model = PegasusForConditionalGeneration.from_pretrained('tanay/efactsum-pegasus-xsum')
    tokenizer = PegasusTokenizer.from_pretrained('tanay/efactsum-pegasus-xsum')

article = ARTICLE_TO_SUMMARIZE.lower()
inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
```

