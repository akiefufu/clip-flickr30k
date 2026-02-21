import csv
import random
from collections import defaultdict

# 定义输入输出路径
caption_file = "data/captions.txt"

train_out = "data/train_captions.txt"
test_out  = "data/test_captions.txt"

# 设定划分比例
split_ratio = 0.8
random.seed(42)

image_caption_dict = defaultdict(list)

with open(caption_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        img = row[0].strip()
        cap = row[1].strip()
        image_caption_dict[img].append(cap)

# 获取 image 列表并打乱
images = list(image_caption_dict.keys())
random.shuffle(images)

# 计算分割点， 划分 Train / Test Image Set
split_point = int(len(images) * split_ratio)

train_imgs = images[:split_point]
test_imgs  = images[split_point:]

print("Train images:", len(train_imgs))
print("Test images :", len(test_imgs))

# 写入新的 split 文件
def write_split(img_list, outfile):
    with open(outfile, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        for img in img_list:
            for cap in image_caption_dict[img]:
                writer.writerow([img, cap])

write_split(train_imgs, train_out)
write_split(test_imgs, test_out)

print("Split Done!")