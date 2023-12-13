import torch
from .HSCNN_Plus import HSCNN_Plus
from .AWAN import AWAN
from .DsTer import DsTer
from .proposed import Main

def model_generator(method, pretrained_model_path=None):
    elif method == 'dster':
        model = DsTer(in_chans=4,out_chans=103,window_size=8, depths=[6, 6, 6, 6],embed_dim=90, num_heads=[6, 6, 6, 6]).cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'awan':
        model = AWAN().cuda()
    elif method == 'proposed':
        model = Main().cuda()
    else:
        print(f'Method {method} is not defined')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
