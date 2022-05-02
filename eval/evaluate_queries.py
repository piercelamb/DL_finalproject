import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def main():
    eval_set = load_dataset('csv', data_files='eval/queries_top100_no13.csv')
    tokenizer = AutoTokenizer.from_pretrained()

    model = AutoModelForCausalLM.from_pretrained('DL_finalproject/models/en_dense_lm_125m/model.pt',)

    inputs = tokenizer(
        eval_set,
        # max_length=max_input_length,
        truncation=True,
        return_tensors='pt',
        padding=True).to('cuda')

    with torch.no_grad():
        results = model.generate(**inputs)
        print(results)


if __name__ == '__main__':
    main()