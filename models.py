import os

# Gets a list of all model paths, e.g. ['./models/en_dense_lm_355m/model.pt', './models/en_dense_lm_125m/model.pt']
def get_model_list(path):
    model_checkpoint_list = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            model_checkpoint_list.append(os.path.join(root, name, 'model.pt'))

    return model_checkpoint_list