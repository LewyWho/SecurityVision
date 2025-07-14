import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class COCODataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, target_size=(640, 640)):
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        self.img_to_info = {}
        for img in self.coco_data['images']:
            self.img_to_info[img['id']] = img
        self.cat_to_info = {}
        for cat in self.coco_data['categories']:
            self.cat_to_info[cat['id']] = cat
        self.ids = list(self.img_to_info.keys())
        self.base_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.img_to_info[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotations = self.img_to_anns.get(img_id, [])
        bboxes = []
        category_ids = []
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            bboxes.append(bbox)
            category_ids.append(category_id)
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
        else:
            transformed = self.base_transform(image=image, bboxes=bboxes, category_ids=category_ids)
        image = transformed['image']
        bboxes = transformed['bboxes']
        category_ids = transformed['category_ids']
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            category_ids = torch.tensor(category_ids, dtype=torch.long)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            category_ids = torch.empty((0,), dtype=torch.long)
        return {
            'image': image,
            'bboxes': bboxes,
            'category_ids': category_ids,
            'image_id': img_id
        }
def get_data_loaders(train_json, test_json, train_dir, test_dir, batch_size=8, num_workers=4):
    train_transform = A.Compose([
        A.Resize(height=640, width=640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    test_transform = A.Compose([
        A.Resize(height=640, width=640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    train_dataset = COCODataset(train_json, train_dir, transform=train_transform)
    test_dataset = COCODataset(test_json, test_dir, transform=test_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader, test_loader
def collate_fn(batch):
    images = []
    bboxes = []
    category_ids = []
    image_ids = []
    for sample in batch:
        images.append(sample['image'])
        bboxes.append(sample['bboxes'])
        category_ids.append(sample['category_ids'])
        image_ids.append(sample['image_id'])
    images = torch.stack(images)
    return {
        'images': images,
        'bboxes': bboxes,
        'category_ids': category_ids,
        'image_ids': image_ids
    }
def visualize_sample(dataset, idx=0):
    sample = dataset[idx]
    image = sample['image'].permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    bboxes = sample['bboxes']
    category_ids = sample['category_ids']
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    category_names = ['creeping', 'crawling', 'stooping', 'climbing', 'other']
    for bbox, cat_id in zip(bboxes, category_ids):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[cat_id-1], facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, category_names[cat_id-1], color=colors[cat_id-1], fontsize=12, weight='bold')
    ax.set_title(f'Image ID: {sample["image_id"]}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(
        'Data/train.json', 
        'Data/test.json', 
        'Data/train', 
        'Data/test',
        batch_size=4
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    for batch in train_loader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Number of bboxes in first image: {len(batch['bboxes'][0])}")
        break 