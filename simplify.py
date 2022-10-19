import torch


def simplify_pth(pth_name):
    checkpoint_dict = torch.load(f"./pth/{pth_name}")
    saved_state_dict = checkpoint_dict['model']
    for i in saved_state_dict.keys():
        saved_state_dict[i] = saved_state_dict[i].half()
    torch.save({'model': saved_state_dict,
                'iteration': None,
                'optimizer': None,
                'learning_rate': None}, f'./pth/half_{pth_name}.pth')


def simplify_hubert(pth_name):
    model = torch.load(f"./pth/{pth_name}")
    for i in model.keys():
        model[i] = model[i].half()
    torch.save(model, f"./pth/half_{pth_name}")


def clean_speaker(pth_name):
    model = torch.load(f"./pth/{pth_name}")
    for i in range(0, len(model['model']['emb_g.weight'])):
        model['model']['emb_g.weight'][i] = 0
    torch.save(model, f"./pth/clean_{pth_name}")


# 模型放在pth文件夹，只输入名字不需要路径，自动改成half_xxx
simplify_pth("243_epochs.pth")
