import torch


def simplify_pth(pth_name):
    model = torch.load(f"./pth/{pth_name}")
    for i in model['model'].keys():
        model['model'][i] = model['model'][i].half()
    torch.save(model, f"./pth/half_{pth_name}")


def simplify_hubert(pth_name):
    model = torch.load(f"./pth/{pth_name}")
    for i in model.keys():
        model[i] = model[i].half()
    torch.save(model, f"./pth/half_{pth_name}")


simplify_hubert("hubert.pt")
