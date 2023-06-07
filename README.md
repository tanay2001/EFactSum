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
Improving factual consistency of abstractive summarization has been a widely studied topic. However, most of the prior works on training factuality-aware models have ignored the negative effect it has on summary quality. We propose EFactSum (i.e., **E**ffective **Fact**ual **Sum**marization), a candidate summary generation and ranking technique to improve summary factuality without sacrificing summary quality. We show that using a contrastive learning framework ([Liu et al. 2022](https://aclanthology.org/2022.acl-long.207.pdf)) with our refined candidate summaries leads to significant gains on both factuality and similarity-based metrics. Specifically, we propose a ranking strategy in which we effectively combine two metrics, thereby preventing any conflict during training. Models trained using our approach show up to 6 points of absolute improvement over the base model with respect to FactCC on XSUM and 11 points on CNN/DM, without negatively affecting either similarity-based metrics or absractiveness.

## Generating data

Alternatively, you can download the data from these links [XSUM](https://drive.google.com/file/d/1v8UReXqlE7_9K2SZe6qG9NSMyOuqSiTI/view?usp=sharing), [CNN/DM](https://drive.google.com/file/d/1Co0cIjQExn6YpG1C8PWcolppZiii7wgi/view?usp=sharing)

## Training

## Decoding


## Model Outputs
We summarize the outputs from our models below

|          | Model |  Source | Model Output | Reference Output |
|----------|---------| --------- | ---------|---------|
| CNNDM    | `tanay/efactsum-bart-cnndm` | [cnndm.test.source](outputs/cnndm.test.source.txt) | [cnndm.test.ours](outputs/cnndm.test.ours.txt) | [cnndm.test.target](outputs/cnndm.test.target.txt)  |
| XSum     | `tanay/efactsum-pegasus-xsum` | [xsum.test.source](outputs/xsum.test.source.txt) |  [xsum.test.ours](outputs/xsum.test.ours.txt) | [xsum.test.target](outputs/xsum.test.target.text)  |



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