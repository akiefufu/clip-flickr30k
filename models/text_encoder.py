from transformers import BertModel
import torch.nn as nn
import torchvision.models as models

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # 调用的是 HuggingFace 实现的 Bert
        # 1. 构建Transformer网络结构
        # 2. 加载预训练权重
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # BERT输出是 768维
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):

        # input_ids : [B , L]
        # attention_mask : [B , L]
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)
        # Bert 输出
        # last_hidden_state.shape = [batch_size , seq_len , 768]

        cls = output.last_hidden_state[:,0,:]
        # 取CLS Token
        # cls = output.last_hidden_state[:,0,:]
        # 整个句子的全局语义表示（sentence embedding）

        return self.projection(cls)
        # 输出形状
        # text_embedding.shape = [batch_size , 512]
