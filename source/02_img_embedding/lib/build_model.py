import torchvision.models as models
from torch import nn
from torch.nn import functional as F


class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x)


def build_model(pretrained):

    model_main = models.resnet34(pretrained=pretrained)
    model_main.fc = nn.Linear(512, 512)

    model = nn.Sequential(model_main, Normalize())

    model.cuda()
    return model
