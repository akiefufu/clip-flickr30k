import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, tokenizer=None):
        self.image_dir = image_dir  # Flickr30k图片目录
        self.transform = transform  # ResNet输入预处理
        self.tokenizer = tokenizer  # BERT tokenizer

        # Flickr30k 每张图片：
        #   对应 5条 caption
        self.samples = []

        # 检查路径
        print("Caption file exists:", os.path.exists(caption_file))
        print("Image dir exists:", os.path.exists(self.image_dir))

        with open(caption_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            # 跳过表头
            next(reader)

            for row in reader:
                # 每一行形如：
                #   1000092795.jpg , "Two young guys..."
                if len(row) < 2:
                    continue

                # 提取字段
                img_name = row[0].strip()
                caption = row[1].strip()

                # 拼接图片路径
                img_path = os.path.join(self.image_dir, img_name)

                # 过滤不存在的图片
                if not os.path.exists(img_path):
                    continue

                # 保存 CLIP 正样本对
                self.samples.append((img_name, caption))

        print("Loaded samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    # getitem() —— 构造 CLIP 输入对
    # 当 DataLoader 需要一个样本时：dataset[idx]
    # 就会调用：__getitem__(idx)
    def __getitem__(self, idx):
        # 取样本
        image_path, caption_text = self.samples[idx]

        # 读取图像， 打开图像
        image = Image.open(os.path.join(self.image_dir, image_path)).convert("RGB")

        # 图像预处理（ResNet要求）
        # 应用 transform
        if self.transform:
            image = self.transform(image)

        # batch = {
        # image            [B,3,224,224]
        # input_ids        [B,77]
        # attention_mask   [B,77]
        # }

        # Caption Tokenization
        if self.tokenizer:
            # BERT tokenizer
            encoding = self.tokenizer(
                caption_text,
                padding="max_length",
                truncation=True,
                max_length=77,  # Batch 内序列必须同一长度
                return_tensors="pt" # 默认 tokenizer 输出：Python list
            )

            # 使用 tokenizer 将 token 字符串转换为整数 token id 序列
            # 去掉 batch 维度，得到形状为 [seq_len] 的张量
            input_ids = encoding["input_ids"].squeeze(0)

            # 使用 attention_mask 来标记哪些 token 是真实的
            attention_mask = encoding["attention_mask"].squeeze(0)

        else:
            # 如果没有 tokenizer，可返回空张量或抛出异常，这里假设 tokenizer 必须提供
            input_ids = None
            attention_mask = None

        # 返回CLIP训练输入格式：
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        # 即一个样本是：
        # Image Tensor
        # Token ID Sequence
        # Mask Sequence

