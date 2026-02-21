import torch
import torch.nn as nn
import torch.nn.functional as F
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder

class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        # 控制 softmax 的“尖锐程度”
        # 如果 τ 很小：softmax更尖锐
        # 如果 τ 很大：softmax更平滑

    def forward(self, image, input_ids, attention_mask):
        image_feat = self.image_encoder(image)
        text_feat = self.text_encoder(input_ids, attention_mask)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logits = image_feat @ text_feat.T / self.temperature
        return logits


def clip_loss(logits):
    labels = torch.arange(logits.size(0)).to(logits.device)

    # 给定 Image i 在所有 Text 中找到正确 caption
    loss_i = F.cross_entropy(logits, labels)

    # 给定 Text j 在所有 Image 中找到正确图片
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2

# 最终学习到跨模态统一语义空间
