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