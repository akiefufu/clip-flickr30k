import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import Flickr30kDataset
from models.clip_model import CLIP
import torch.nn.functional as F

import torchvision.transforms as transforms
from transformers import BertTokenizer


# -----------------------------
# CLIP Loss
# -----------------------------
def clip_loss(logits):
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

# -----------------------------
# Train Function
# -----------------------------
def train():

    # =========================
    # 超参数
    # =========================
    image_dir = "data/images"
    caption_file = "data/train_captions.txt"

    batch_size = 32
    epochs = 10
    lr = 1e-4
    embed_dim = 512     # 图像和文本最终投影到的公共空间维度
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Dataset & DataLoader
    # =========================
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Tokenizer 加载
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = Flickr30kDataset(
        image_dir=image_dir,
        caption_file=caption_file,
        transform=transform,
        tokenizer=tokenizer
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    # DataLoader会自动调用：__getitem__()
    # 32 次，然后拼接成：
    # batch =
    # {
    #  "image":          [32 , 3 , 224 , 224]
    #  "input_ids":      [32 , 77]
    #  "attention_mask": [32 , 77]
    # }


    # =========================
    # Model
    # =========================
    model = CLIP(embed_dim=embed_dim).to(device)

    # =========================
    # Optimizer
    # =========================
    # 必须使用 AdamW 而不是 Adam
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # model.parameters() 包含：
    # ResNet 参数
    # BERT 参数
    # projection layer 参数

    # 学习率变化：
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # 整个 CLIP 训练代码里 保证 FP16 混合精度训练不会崩溃的核心稳定器
    scaler = GradScaler()

    # =========================
    # 训练循环
    # =========================
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        # 把你的 DataLoader 训练迭代器包装成一个带有进度监控能力的可视化迭代器（Progress Bar Iterator），
        # 用于实时追踪训练过程中的 batch 级进度与状态指标
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:

            # batch["image"] 通常是一个 4D Tensor：[B,C,H,W]
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            # AMP 混合精度训练
            with autocast():
                logits = model(images, input_ids, attention_mask)
                loss = clip_loss(logits)

            # 标准 AMP 训练模板
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # 是在 动态更新 tqdm 进度条右侧的实时训练指标（metrics）显示区域，
            # 用于在每一个 batch 结束后，把当前 loss 数值附加到进度条上
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # =========================
        # 保存模型
        # =========================
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"checkpoints/clip_epoch_{epoch+1}.pth"
        )

    print("Training Finished!")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    train()
