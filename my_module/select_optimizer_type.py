import torch

def select_optimizer_type(optimzer_type,model,learning_rate):
    if optimzer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),learning_rate) 
        return optimizer

    elif optimzer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),learning_rate,weight_decay)
        return optimizer

    else:
        print("select optimzer_type")
