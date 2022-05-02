import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Features, Value, ClassLabel, Sequence

TOKENIZER = AutoTokenizer.from_pretrained('gpt2')
MODEL = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")

def main():
    eval_set = load_dataset('csv', data_files='queries_top100_no13.csv')['train']
    # eval set is a Dataset with columns Query and Answer after this
    few_shot = ""
    for idx, row in enumerate(eval_set):
        question = str(row['Query'])
        answer = str(row['Answer'])
        combined = question + answer + '\n'
        if (idx == 0) or (idx % 10 != 0):
            few_shot += " " + combined
        else:
            few_shot += question
            tokenized = TOKENIZER.encode(few_shot, return_tensors='pt')
            greedy_output = MODEL.generate(tokenized, max_length=200)
            print(TOKENIZER.decode(greedy_output[0], skip_special_tokens=True))
            few_shot = ""


    # with torch.no_grad():
    #     results = model.generate(**inputs)
    #     print(results)


if __name__ == '__main__':
    main()