from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor


# However, we can use LayoutLMv2Processor to easily prepare the data for the model.
# We give a document image as input to the processor, and it will create input_ids,
# attention_mask, token_type_ids and bbox for us. Internally, it will apply PyTesseract
# to get the words and bounding boxes, it will normalize the bounding boxes according
# to the size of the image, and it will turn everything into token-level inputs.
# It will also resize the document image to 224x224, as the model also requires an image input.

#Btw, if you prefer to use your own OCR engine, you still can. In that case, you can
# provide your own words and (normalized) bounding boxes to the processor.

# refer to this link for discussion of how to use your own OCR
# https://huggingface.co/docs/transformers/model_doc/layoutlmv2#usage-layoutlmv2processor
def get_processor():
    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)
    return processor

def test_inputs(image, processor):
    encoded_inputs = processor(image, return_tensors="pt")
    print(processor.tokenizer.decode(encoded_inputs.input_ids.squeeze().tolist()))


