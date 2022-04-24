from transformers import LayoutLMv2ForSequenceClassification
from data_preprocessor import get_encoded_data, get_labeled_data, ID2LABEL, PROCESSOR
from utils.data import get_image
from processor import get_processor
from tqdm.notebook import tqdm
import torch

def train(dataloader, model, optimizer, num_train_epochs, data_len):
    global_step = 0
    t_total = len(dataloader) * num_train_epochs  # total number of training steps

    # put the model in training mode
    model.train()
    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        running_loss = 0.0
        correct = 0
        for batch in tqdm(dataloader): #TODO get tqdm working right
            # forward pass
            outputs = model(**batch)
            loss = outputs.loss

            running_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch['labels']).float().sum()

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / data_len
        print("Training accuracy:", accuracy.item())

def test(image, processor):
    # prepare image for the model
    encoded_inputs = processor(image, return_tensors="pt")

    # make sure all keys of encoded_inputs are on the same device as the model
    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(model.device)

    # forward pass
    outputs = model(**encoded_inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", ID2LABEL[predicted_class_idx])

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb
if __name__ == "__main__":
    batch_size = 4
    learning_rate = 0.00005
    epochs = 10
    # get encoded, batched data
    data, labels = get_labeled_data()
    encoded_dataset = get_encoded_data(data, labels)
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size)

    # define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights initialized from the pretrained model
    # classification layer initialized randomly and will be fine-tuned
    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",
                                                                num_labels=len(labels))
    model.to(device)

    # train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train(dataloader, model, optimizer, epochs, len(data))

    # test the model
    image = get_image("RVL_CDIP_one_example_per_class/resume/0000157402.tif", "resume")
    test(image, PROCESSOR)


# TODO
# - Get auditor annotated data from prod
# - Get PDFs, raw_text & textract_response from prod