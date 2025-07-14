import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from tqdm import tqdm
from dataset import COCODataset
from models import get_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
class ModelEvaluator:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        albumentations_transform = A.Compose([
            A.Resize(height=config['image_size'], width=config['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        self.test_dataset = COCODataset(
            json_file=config['test_ann'],
            img_dir=os.path.join(config['data_root'], 'test'),
            transform=albumentations_transform
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn
        )
        self.class_names = ['creeping', 'crawling', 'stooping', 'climbing', 'other']
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_name = self.model_path.split('/')[1]
        self.model = get_model(model_name, num_classes=self.config['num_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Модель загружена: {self.model_path}")
    def collate_fn(self, batch):
        images = []
        targets = []
        for sample in batch:
            if isinstance(sample, (list, tuple)) and len(sample) == 2:
                image, target = sample
                images.append(image)
                targets.append(target)
            elif isinstance(sample, dict):
                images.append(sample['image'])
                targets.append({
                    'bboxes': sample['bboxes'],
                    'labels': sample['category_ids'],
                    'image_id': sample.get('image_id', 0)
                })
            else:
                raise ValueError(f"Неожиданный формат данных: {type(sample)}")
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
        elif isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if outputs.dim() == 5:
            outputs = outputs[..., -self.config['num_classes']:]
            outputs = outputs.mean(dim=(1, 2, 3))
        elif outputs.dim() == 4:
            outputs = outputs.mean(dim=(2, 3))
        return outputs
    def evaluate_model(self):
        print("Начинаем оценку модели...")
        all_predictions = []
        all_targets = []
        all_probabilities = []
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Оценка"):
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
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        cm = confusion_matrix(all_targets, all_predictions)
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    def plot_confusion_matrix(self, cm, save_path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Истинные классы')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена: {save_path}")
    def plot_class_accuracy(self, report, save_path):
        classes = list(report.keys())[:-3]
        precisions = [report[cls]['precision'] for cls in classes]
        recalls = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        x = np.arange(len(classes))
        width = 0.25
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        plt.xlabel('Классы')
        plt.ylabel('Метрики')
        plt.title('Метрики по классам')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График метрик по классам сохранен: {save_path}")
    def analyze_predictions(self, predictions, targets, probabilities, save_path):
        max_probs = np.max(probabilities, axis=1)
        correct_predictions = np.array(predictions) == np.array(targets)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(max_probs[correct_predictions], alpha=0.7, label='Правильные', bins=20)
        plt.hist(max_probs[~correct_predictions], alpha=0.7, label='Неправильные', bins=20)
        plt.xlabel('Уверенность модели')
        plt.ylabel('Количество предсказаний')
        plt.title('Распределение уверенности')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 3, 2)
        confidence_bins = np.linspace(0, 1, 11)
        accuracies = []
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                acc = np.mean(correct_predictions[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        plt.plot(confidence_bins[:-1], accuracies, 'o-')
        plt.xlabel('Уверенность модели')
        plt.ylabel('Точность')
        plt.title('Точность vs Уверенность')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 3, 3)
        error_counts = {}
        for pred, target in zip(predictions, targets):
            if pred != target:
                pred_class = self.class_names[pred]
                if pred_class not in error_counts:
                    error_counts[pred_class] = 0
                error_counts[pred_class] += 1
        if error_counts:
            classes = list(error_counts.keys())
            counts = list(error_counts.values())
            plt.bar(classes, counts)
            plt.xlabel('Классы')
            plt.ylabel('Количество ошибок')
            plt.title('Распределение ошибок по классам')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Анализ предсказаний сохранен: {save_path}")
def find_model_files():
    import glob
    model_files = glob.glob('models/*/best.pth')
    return model_files
def main():
    parser = argparse.ArgumentParser(description='Оценка обученных моделей')
    parser.add_argument('--model_path', type=str, default=None, help='Путь к обученной модели')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--image_size', type=int, default=512, help='Размер изображения')
    parser.add_argument('--evaluate_all', action='store_true', help='Оценить все найденные модели')
    args = parser.parse_args()
    config = {
        'data_root': 'Data',
        'test_ann': 'Data/test.json',
        'num_classes': 5,
        'batch_size': args.batch_size,
        'image_size': args.image_size
    }
    if args.evaluate_all:
        model_files = find_model_files()
        for model_path in model_files:
            print(f"\nОценка модели: {model_path}")
            evaluator = ModelEvaluator(model_path, config)
            results = evaluator.evaluate_model()
            save_dir = f"evaluation_results/{model_path.split('/')[1]}"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/evaluation_report.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            evaluator.plot_confusion_matrix(results['confusion_matrix'], f"{save_dir}/confusion_matrix.png")
            evaluator.plot_class_accuracy(results['classification_report'], f"{save_dir}/class_metrics.png")
            evaluator.analyze_predictions(results['predictions'], results['targets'], results['probabilities'], f"{save_dir}/prediction_analysis.png")
    else:
        if not args.model_path:
            print("Не указан путь к модели")
            return
        evaluator = ModelEvaluator(args.model_path, config)
        results = evaluator.evaluate_model()
        save_dir = f"evaluation_results/{args.model_path.split('/')[1]}"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/evaluation_report.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        evaluator.plot_confusion_matrix(results['confusion_matrix'], f"{save_dir}/confusion_matrix.png")
        evaluator.plot_class_accuracy(results['classification_report'], f"{save_dir}/class_metrics.png")
        evaluator.analyze_predictions(results['predictions'], results['targets'], results['probabilities'], f"{save_dir}/prediction_analysis.png")
if __name__ == "__main__":
    main() 