import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Features, Value, ClassLabel, Sequence

TOKENIZER = AutoTokenizer.from_pretrained('gpt2')
MODEL = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")

def tokenize_data(row):
    question = str(row['Query'])
    answer = str(row['Answer'])
    tokenized = TOKENIZER.encode(question, return_tensors='pt')
    greedy_output = MODEL.generate(tokenized)
    print(TOKENIZER.decode(greedy_output[0], skip_special_tokens=True))
    return None


def main():
    eval_set = load_dataset('csv', data_files='queries_top100_no13.csv')['train']
    # eval set is a Dataset with columns Query and Answer after this

    inputs = eval_set.map(
        tokenize_data,
        remove_columns=eval_set.column_names,
    )


    # with torch.no_grad():
    #     results = model.generate(**inputs)
    #     print(results)


if __name__ == '__main__':
    main()