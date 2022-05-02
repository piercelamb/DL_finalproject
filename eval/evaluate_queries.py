import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Features, Value, ClassLabel, Sequence

TOKENIZER = AutoTokenizer.from_pretrained('gpt2')
MODEL = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")

def tokenize_data(row, idx):
    question = str(row['Query'])
    answer = str(row['Answer'])
    combined = question + answer
    if idx % 10 == 0:
        # we're at a multiple of ten, call generate
        tokenized = TOKENIZER.encode(question, return_tensors='pt')
        # pass to the model here
        greedy_output = MODEL.generate(tokenized)

        print(TOKENIZER.decode(greedy_output[0], skip_special_tokens=True))
    else:
        tokenized = TOKENIZER.encode(combined, return_tensors='pt')
        # pass to the model here
        # greedy_output = MODEL.generate(tokenized)

    return None


def main():
    eval_set = load_dataset('csv', data_files='queries_top100_no13.csv')['train']
    # eval set is a Dataset with columns Query and Answer after this
    INDEX = 0
    inputs = eval_set.map(
        tokenize_data,
        remove_columns=eval_set.column_names,
        with_indices=True
    )


    # with torch.no_grad():
    #     results = model.generate(**inputs)
    #     print(results)


if __name__ == '__main__':
    main()