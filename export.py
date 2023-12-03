import torch
import vision_transformer as vits
from vision_transformer import DINOHead
from utils import MultiCropWrapper


if __name__ == "__main__":
    teacher = vits.__dict__["vit_small"](patch_size=16)
    model = MultiCropWrapper(teacher, DINOHead(384, 65536))
    x = torch.randn(1, 3, 224, 768)
    try:
        torch.export.export(model, (x, ))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e
