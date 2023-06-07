from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration
import torch

import argparse
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type = str)   
    parser.add_argument('--model_path',type = str)   
    parser.add_argument('--max_length', type = int)
    parser.add_argument('--output_dir', type = str)    
    parser.add_argument('--gen_max_len', type = int)
    parser.add_argument('--gen_min_len', type = int)
    parser.add_argument('--num_beams', type = int)   
    parser.add_argument('--batch_size', type = int)     
    parser.add_argument('--length_penalty', type = float)
    args = parser.parse_args()

    with open(args.source) as f:
        sources = [line.strip() for line in f]

    tokenizer = PegasusTokenizerFast.from_pretrained(args.model_path)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_path)
    model.cuda()
    model.eval()

    print('Model loaded')

    os.makedirs(args.output_dir, exist_ok=True)
    f = open(os.path.join(args.output_dir, 'generations.txt'), 'w')
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(sources), args.batch_size)):
            batch_sources = sources[batch_start:batch_start+args.batch_size]
            inputs = tokenizer(batch_sources, return_tensors="pt", truncation=True, max_length=args.max_length, padding=True).to('cuda')
            outputs = model.generate(**inputs,
                            max_length=args.gen_max_len + 2,  
                            min_length=args.gen_min_len + 1, 
                            no_repeat_ngram_size=3,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty
                            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for summary in decoded:
                f.write(summary + '\n')


if __name__ == '__main__':
    main()
