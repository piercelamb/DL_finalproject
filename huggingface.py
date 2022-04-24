from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")

    input_ids = tokenizer.encode("I enjoy walking with my cute dog", return_tensors='pt')

    greedy_output = model.generate(input_ids, max_length=50)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))