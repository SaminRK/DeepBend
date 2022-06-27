def get_model(model_name):
    if model_name == "model35":
        from .model35 import nn_model as model35

        return model35
