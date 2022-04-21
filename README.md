## Run instructions:

- Clone the repository.
- In the root repository, if they aren't there, create a `/data` folder, `/data_bin` folder, `/checkpoint` folder and `/models` folder
- In the `/data` folder you just created, download and unzip the [deepmind math dataset](https://console.cloud.google.com/storage/browser/_details/mathematics-dataset/mathematics_dataset-v1.0.tar.gz;tab=live_object)
  - File path should look like `/data/mathematics_dataset-v1.0`
- Download the `encoder.json` file and `vocab.bpe` required by the bpe encoder. Make sure they are in the `/data` folder:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
- Download an extract at least one fairseq moe_lm model to the /models folder. For e.g. [this one](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_125m.tar.gz)
  - File path should look like `'/models/en_dense_lm_125m/model.pt'`
- If you've run it at all unsuccessfully before, make sure to delete files it produces as it checks for their existence before creating them.

## Notes

- When I ran preprocess_data(), it produced files in `/data_bin` that looked like:
  - ```
    train.None-None.bin
    train.None-None.idx
    valid.None-None.bin
    valid.None-None.idx
    etc
    ```
  - And I do not know why the `None-None` was included. In order to get fairseq-train to run, I had to manually delete the `None-None`'s 