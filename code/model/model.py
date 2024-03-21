import torch.nn as nn
from torchvision.models import vgg11, vgg11_bn, vgg13
from torchvision.models import resnet18

class vgg_base(nn.Module):
    def __init__(self, input_dim):
        super(vgg_base,self).__init__()
        self.vggmodel=vgg11(pretrained=False).features
        self.vggmodel[0]=nn.Conv2d(input_dim,64,kernel_size = 3, padding= 1)

    def forward(self, x):
        x = self.vggmodel(x)
        return x

class vggbn_base(nn.Module):
    def __init__(self, input_dim):
        super(vggbn_base,self).__init__()
        self.vggmodel=vgg11_bn(pretrained=False).features
        self.vggmodel[0]=nn.Conv2d(input_dim,64,kernel_size = 3, padding= 1)

    def forward(self, x):
        x = self.vggmodel(x)
        return x


class resnet_base(nn.Module):
    def __init__(self, input_dim):
        super(resnet_base,self).__init__()
        self.resnetmodel=resnet18(pretrained=False)
        self.resnetmodel.conv1=nn.Conv2d(input_dim,64,kernel_size = 7, stride=2,padding= 3,bias=False)

    def forward(self, x):
        x = self.resnetmodel(x)
        return x

class My_model(nn.Module):
    def __init__(self, input_dim=1, num_classes=93, model_base="vgg"):
        super(My_model,self).__init__()
        if model_base == "vgg":
            self.backbone=vgg_base(input_dim)
        elif model_base == "vggbn":
            self.backbone=vggbn_base(input_dim)
        elif model_base =="resnet":
            self.backbone=resnet_base(input_dim)

        self.model_base=model_base
        self.avgpool = nn.AvgPool1d(kernel_size=200, stride=1)
        self.linear = nn.Linear(in_features=512, out_features=num_classes)
        self.linear2 = nn.Linear(in_features=1000, out_features=num_classes)
        self.activate = nn.Softmax(dim=1)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, input, label=None):
        result = self.backbone(input)
        if self.model_base in ["vgg","vggbn"]:
            result = result.view(result.size(0), result.size(1), -1)
            result = self.avgpool(result)
            result = result.reshape(result.size(0), -1)
            result = self.linear(result)

        elif self.model_base == "resnet":
            result = self.linear2(result)

        result = self.activate(result)

        _, pred_label = result.max(-1)

        if label is not None: # train
            loss = self.criteria(result, label.view(-1))
            return loss, result, pred_label
        else: # test
            return result, pred_label



