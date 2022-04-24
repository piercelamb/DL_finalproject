import pandas as pd
import os
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from PIL import Image
from processor import get_processor

DATASET_PATH = "RVL_CDIP_one_example_per_class"
RAW_LABELS = [label for label in os.listdir(DATASET_PATH)]
ID2LABEL = {v: k for v, k in enumerate(RAW_LABELS)}
LABEL2ID = {k: v for v, k in enumerate(RAW_LABELS)}
PROCESSOR = get_processor()

# TODO redo this piece as one big function like the ipynb file and see if it works

def get_labeled_data():
    images = []
    labels = []

    for label_folder, _, file_names in os.walk(DATASET_PATH):
        if label_folder != DATASET_PATH:
            label = label_folder[31:]
            for _, _, image_names in os.walk(label_folder):
                relative_image_names = []
                for image_file in image_names:
                    relative_image_names.append(DATASET_PATH + "/" + label + "/" + image_file)
                images.extend(relative_image_names)
                labels.extend([label] * len(relative_image_names))

    data = pd.DataFrame.from_dict({'image_path': images, 'label': labels})
    return data, labels


def preprocess_data(examples):

    # take a batch of images
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]

    encoded_inputs = PROCESSOR(images, padding="max_length", truncation=True)

    # add labels
    encoded_inputs["labels"] = [LABEL2ID[label] for label in examples["label"]]

    return encoded_inputs

def get_encoded_data(data, labels):
    dataset = Dataset.from_pandas(data)

    features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': ClassLabel(num_classes=len(labels), names=labels),
    })

    encoded_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names, features=features,
                                  batched=True, batch_size=2)

    encoded_dataset.set_format(type="torch")
    return encoded_dataset