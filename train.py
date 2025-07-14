import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from dataset import COCODataset
from models import get_model

class Trainer:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            device_name = str(device)
            print(f"Используется устройство: {device_name} (TPU/XLA)")
        except ImportError:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Используется устройство: {device} (CUDA)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                print(f"Используется устройство: {device} (MPS)")
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device('xpu')
                print(f"Используется устройство: {device} (XPU)")
            else:
                device = torch.device('cpu')
                print(f"Используется устройство: {device} (CPU)")
        self.device = device
        self.model = get_model(model_name, num_classes=config['num_classes'])
        self.model.to(self.device)
        albumentations_transform = A.Compose([
            A.Resize(height=config['image_size'], width=config['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        self.train_dataset = COCODataset(
            json_file=config['train_ann'],
            img_dir=os.path.join(config['data_root'], 'train'),
            transform=albumentations_transform
        )
        self.val_dataset = COCODataset(
            json_file=config['test_ann'],
            img_dir=os.path.join(config['data_root'], 'test'),
            transform=albumentations_transform
        )
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=4,
            collate_fn=self.collate_fn
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs']
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.is_detection = True
    def collate_fn(self, batch):
        images = []
        targets = []
        for sample in batch:
            images.append(sample['image'])
            targets.append({
                'bboxes': sample['bboxes'],
                'labels': sample['category_ids'],
                'image_id': sample['image_id']
            })
        images = torch.stack(images)
        return images, targets
    def process_model_outputs(self, outputs):
        if isinstance(outputs, dict):
            if 'classification' in outputs:
                cls_outputs = outputs['classification']
                if isinstance(cls_outputs, list):
                    outputs = cls_outputs[0]
                else:
                    outputs = cls_outputs
            elif 'confidence' in outputs:
                outputs = outputs['confidence']
                if isinstance(outputs, list):
                    outputs = outputs[0]
            elif 'roi' in outputs:
                outputs = outputs['roi']
            elif 'cls_outputs' in outputs:
                outputs = outputs['cls_outputs'][0]
        elif isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if outputs.dim() == 5:
            outputs = outputs[..., -self.config['num_classes']:]
            outputs = outputs.mean(dim=(1, 2, 3))
        elif outputs.dim() == 4:
            outputs = outputs.mean(dim=(2, 3))
        elif outputs.dim() == 2:
            pass
        else:
            raise ValueError(f"Неожиданная размерность выходов: {outputs.shape}")
        return outputs
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(self.train_loader, desc=f"Обучение {self.model_name}")
        for batch_idx, (images, targets) in enumerate(progress_bar):
            if self.is_detection:
                images = [img.to(self.device) for img in images]
                batch_targets = []
                for target in targets:
                    batch_targets.append({
                        'boxes': target['bboxes'].to(self.device),
                        'labels': target['labels'].to(self.device)
                    })
                self.optimizer.zero_grad()
                loss_dict = self.model(images, batch_targets)
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            else:
                images = images.to(self.device)
                batch_targets = []
                for target in targets:
                    if len(target['labels']) > 0:
                        original_label = target['labels'][0].item()
                        mapped_label = original_label - 1
                        batch_targets.append(mapped_label)
                    else:
                        batch_targets.append(0)
                batch_targets = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                outputs = self.process_model_outputs(outputs)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == batch_targets).sum().item()
                total_predictions += batch_targets.size(0)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
                })
        if self.is_detection:
            avg_loss = total_loss / len(self.train_loader)
            return avg_loss, None
        else:
            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100 * correct_predictions / total_predictions
            return avg_loss, accuracy
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Валидация {self.model_name}")
            for batch_idx, (images, targets) in enumerate(progress_bar):
                if self.is_detection:
                    images = [img.to(self.device) for img in images]
                    batch_targets = []
                    for target in targets:
                        batch_targets.append({
                            'boxes': target['bboxes'].to(self.device),
                            'labels': target['labels'].to(self.device)
                        })
                    loss_dict = self.model(images, batch_targets)
                    loss = sum(loss for loss in loss_dict.values())
                    total_loss += loss.item()
                    progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                else:
                    images = images.to(self.device)
                    batch_targets = []
                    for target in targets:
                        if len(target['labels']) > 0:
                            original_label = target['labels'][0].item()
                            mapped_label = original_label - 1
                            batch_targets.append(mapped_label)
                        else:
                            batch_targets.append(0)
                    batch_targets = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                    outputs = self.model(images)
                    outputs = self.process_model_outputs(outputs)
                    loss = self.criterion(outputs, batch_targets)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == batch_targets).sum().item()
                    total_predictions += batch_targets.size(0)
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
                    })
        if self.is_detection:
            avg_loss = total_loss / len(self.val_loader)
            return avg_loss, None
        else:
            avg_loss = total_loss / len(self.val_loader)
            accuracy = 100 * correct_predictions / total_predictions
            return avg_loss, accuracy
    def train(self):
        best_val_accuracy = 0
        best_model_state = None
        for epoch in range(self.config['epochs']):
            print(f"\nЭпоха {epoch+1}/{self.config['epochs']}")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            if val_acc is not None and val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict()
            self.scheduler.step()
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if train_acc is not None:
                print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        final_train_accuracy = self.train_accuracies[-1] if self.train_accuracies else None
        final_val_accuracy = self.val_accuracies[-1] if self.val_accuracies else None
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.save_model('best')
        self.save_model('final')
        self.plot_training_curves()
        return {
            'model_name': self.model_name,
            'best_val_accuracy': best_val_accuracy,
            'final_train_accuracy': final_train_accuracy,
            'final_val_accuracy': final_val_accuracy
        }
    def save_model(self, suffix):
        os.makedirs(f"models/{self.model_name}", exist_ok=True)
        save_path = f"models/{self.model_name}/{suffix}.pth"
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        print(f"Модель сохранена: {save_path}")
    def plot_training_curves(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Acc')
        plt.plot(epochs, self.val_accuracies, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"models/{self.model_name}/training_curves.png")
        plt.close()
def main():
    parser = argparse.ArgumentParser(description='Обучение моделей детекции объектов')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['yolo', 'efficientdet', 'ssd', 'all'],
                       help='Модель для обучения')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512, help='Размер изображения')
    args = parser.parse_args()
    config = {
        'data_root': 'Data',
        'train_ann': 'Data/train.json',
        'test_ann': 'Data/test.json',
        'num_classes': 5,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 0.0001,
        'image_size': args.image_size
    }
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['yolo', 'efficientdet', 'ssd']
    else:
        models_to_train = [args.model]
    results = []
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"ОБУЧЕНИЕ МОДЕЛИ: {model_name.upper()}")
        print(f"{'='*50}")
        try:
            trainer = Trainer(model_name, config)
            result = trainer.train()
            results.append(result)
            print(f"\nРезультаты для {model_name}:")
            print(f"Лучшая валидационная точность: {result['best_val_accuracy']:.2f}%")
            print(f"Финальная обучающая точность: {result['final_train_accuracy']:.2f}%")
            print(f"Финальная валидационная точность: {result['final_val_accuracy']:.2f}%")
        except Exception as e:
            print(f"Ошибка при обучении модели {model_name}: {str(e)}")
            continue
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"training_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*50}")
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"{'='*50}")
        results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['model_name']}: {result['best_val_accuracy']:.2f}%")
        print(f"\nРезультаты сохранены в: {results_file}")
if __name__ == "__main__":
    main() 