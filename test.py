import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer

from dataset import Flickr30kDataset
from models.clip_model import CLIP


# =========================
# Evaluation Function
# =========================
def evaluate_clip(model, dataloader, device="cuda"):

    # 进入推理模式
    model.eval()

    i2t_top1 = 0
    i2t_top5 = 0
    t2i_top1 = 0
    t2i_top5 = 0
    total = 0

    # 禁止构建计算图
    with torch.no_grad():
        for batch in dataloader:

            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(images, input_ids, attention_mask)
            labels = torch.arange(logits.size(0)).to(device)

            # -------------------------
            # Image → Text Retrieval
            # -------------------------
            pred_i2t = torch.argmax(logits, dim=1)
            i2t_top1 += (pred_i2t == labels).sum().item()

            # 对于每张图片，找最相似的5个文本
            top5_i2t = torch.topk(logits, k=5, dim=1).indices
            for i in range(len(labels)):
                if labels[i] in top5_i2t[i]:
                    i2t_top5 += 1

            # -------------------------
            # Text → Image Retrieval
            # -------------------------
            pred_t2i = torch.argmax(logits.T, dim=1)
            t2i_top1 += (pred_t2i == labels).sum().item()

            top5_t2i = torch.topk(logits.T, k=5, dim=1).indices
            for i in range(len(labels)):
                if labels[i] in top5_t2i[i]:
                    t2i_top5 += 1

            total += images.size(0)

    # 最终指标计算
    print("\n========== Evaluation Result ==========")
    print(f"Image → Text Top-1 Accuracy : {i2t_top1/total:.4f}")
    print(f"Image → Text Top-5 Accuracy : {i2t_top5/total:.4f}")
    print(f"Text  → Image Top-1 Accuracy : {t2i_top1/total:.4f}")
    print(f"Text  → Image Top-5 Accuracy : {t2i_top5/total:.4f}")
    print("=======================================\n")


# =========================
# Main Test Pipeline
# =========================
def test():

    # -------------------------
    # Config
    # -------------------------
    image_dir = "data/images"
    caption_file = "data/test_captions.txt"

    batch_size = 32
    embed_dim = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Transform (必须和训练一致)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

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
        shuffle=False
    )

    # -------------------------
    # Load Trained Model
    # -------------------------
    model = CLIP(embed_dim=embed_dim).to(device)

    # 加载训练好的CLIP模型
    model.load_state_dict(
        torch.load("checkpoints/clip_epoch_10.pth",
                   map_location=device)
    )

    print("Model loaded successfully!")

    # -------------------------
    # Evaluate
    # -------------------------
    evaluate_clip(model, dataloader, device)


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    test()