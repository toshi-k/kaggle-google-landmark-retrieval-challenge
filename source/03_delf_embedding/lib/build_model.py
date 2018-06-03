import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x)


class ModelMain(nn.Module):

    def __init__(self):
        super(ModelMain, self).__init__()
        self.l1 = nn.Linear(40, 128)
        self.l2 = nn.Linear(128, 192)
        self.f1 = nn.Linear(192, 512)
        self.f2 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(192)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):

        batch_size = len(x)

        h0 = x.transpose(1, 2).contiguous().view((-1, 40))
        h1 = self.bn1(F.leaky_relu(self.l1(h0)))
        h2 = self.bn2(F.leaky_relu(self.l2(h1)))

        he = h2.view((batch_size, -1, 192))
        hs = he.sum(1)

        j1 = self.bn3(F.leaky_relu(self.f1(hs)))
        j2 = self.f2(j1)

        return j2


def build_model():

    model = nn.Sequential(ModelMain(), Normalize())

    model.cuda()
    return model


def debug():

    model = build_model()

    input_tensor = torch.FloatTensor(2, 40, 1000).zero_().cuda()
    output_tensor = model.forward(Variable(input_tensor))
    print(output_tensor.shape)


if __name__ == '__main__':
    debug()
