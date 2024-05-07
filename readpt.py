import torch

model=torch.load("./experiment/best_valid_ckpt.pt")
print(model)
#model.eval()