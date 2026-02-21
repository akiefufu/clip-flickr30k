import torch.nn as nn
import torchvision.models as models

# ResNet18
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # ResNet18 是在 ImageNet 上训练好的 CNN
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        # 去掉分类头
        self.feature = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(512, embed_dim)

    def forward(self, x):

        x = self.feature(x)
        # 输出变成
        # (B , 512 , 1 , 1)

        # flatten 成向量
        x = x.flatten(1)

        x = self.projection(x)
        return x
        # 输出形状
        # image_embedding.shape = [batch_size , 512]
