import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import argparse
import json
from tqdm import tqdm
from models import get_model
class ObjectDetector:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.class_names = ['creeping', 'crawling', 'stooping', 'climbing', 'other']
        self.class_colors = {
            'creeping': 'red',
            'crawling': 'orange', 
            'stooping': 'yellow',
            'climbing': 'green',
            'other': 'blue'
        }
        self.transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_name = self.model_path.split('/')[1]
        self.model = get_model(model_name, num_classes=self.config['num_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Модель загружена: {self.model_path}")
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return input_tensor, image, original_size
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
    def detect_objects(self, image_path, confidence_threshold=0.5):
        input_tensor, original_image, original_size = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            outputs = self.process_model_outputs(outputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, 1)
            valid_detections = confidence > confidence_threshold
            results = []
            for i in range(len(predictions)):
                if valid_detections[i]:
                    class_id = predictions[i].item()
                    class_name = self.class_names[class_id]
                    conf = confidence[i].item()
                    original_class_id = class_id + 1
                    results.append({
                        'class_id': original_class_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': [0, 0, original_size[0], original_size[1]]
                    })
        return results, original_image, original_size
    def visualize_detections(self, image, detections, save_path=None):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            color = self.class_colors.get(class_name, 'gray')
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            x_text = bbox[0] + (bbox[2] - bbox[0]) / 2
            y_text = bbox[1] - 10
            text = f"{class_name}"
            ax.text(x_text, y_text, text, fontsize=14, color=color, weight='bold', ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            ax.text(x_text, y_text + 16, f"{confidence:.2f}", fontsize=10, color=color, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.5))
        ax.set_title('Детекция объектов', fontsize=16, weight='bold')
        ax.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Результат сохранен: {save_path}")
        else:
            plt.show()
        plt.close()
    def process_single_image(self, image_path, confidence_threshold=0.5, save_dir=None):
        print(f"Обработка изображения: {image_path}")
        detections, image, original_size = self.detect_objects(image_path, confidence_threshold)
        print(f"Найдено объектов: {len(detections)}")
        for detection in detections:
            print(f"  - {detection['class_name']}: {detection['confidence']:.3f}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            save_path = os.path.join(save_dir, f"detected_{filename}")
            self.visualize_detections(image, detections, save_path)
        return detections
    def process_directory(self, input_dir, confidence_threshold=0.5, save_dir=None):
        import random
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_dir, file))
        print(f"Найдено {len(image_files)} изображений для обработки")
        if len(image_files) > 40:
            image_files = random.sample(image_files, 40)
            print(f"Выбрано случайных 40 изображений для инференса")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        all_results = {}
        for image_path in tqdm(image_files, desc="Обработка изображений"):
            try:
                detections = self.process_single_image(
                    image_path, confidence_threshold, save_dir
                )
                all_results[image_path] = detections
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {str(e)}")
                continue
        return all_results
    def generate_summary_report(self, results, save_path):
        class_counts = {}
        class_confidences = {}
        for image_path, detections in results.items():
            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    class_confidences[class_name] = []
                class_counts[class_name] += 1
                class_confidences[class_name].append(confidence)
        avg_confidences = {}
        for class_name, confidences in class_confidences.items():
            avg_confidences[class_name] = np.mean(confidences)
        report = {
            'total_images': len(results),
            'total_detections': sum(len(detections) for detections in results.values()),
            'class_statistics': {
                class_name: {
                    'count': class_counts.get(class_name, 0),
                    'avg_confidence': avg_confidences.get(class_name, 0)
                }
                for class_name in self.class_names
            },
            'detections_per_image': {
                image_path: len(detections) 
                for image_path, detections in results.items()
            }
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Сводный отчет сохранен: {save_path}")
        print(f"\n{'='*50}")
        print("СВОДНАЯ СТАТИСТИКА")
        print(f"{'='*50}")
        print(f"Всего изображений: {report['total_images']}")
        print(f"Всего детекций: {report['total_detections']}")
def main():
    parser = argparse.ArgumentParser(description='Инференс обученных моделей')
    parser.add_argument('--model_path', type=str, required=True, help='Путь к обученной модели')
    parser.add_argument('--input', type=str, required=True, help='Путь к изображению или директории')
    parser.add_argument('--confidence', type=float, default=0.5, help='Порог уверенности для детекции')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Директория для сохранения результатов')
    parser.add_argument('--image_size', type=int, default=512, help='Размер изображения для модели')
    args = parser.parse_args()
    config = {
        'data_root': 'Data',
        'train_ann': 'Data/train.json',
        'test_ann': 'Data/test.json',
        'num_classes': 5,
        'batch_size': 1,
        'image_size': args.image_size
    }
    if not os.path.exists(args.model_path):
        print(f"Модель не найдена: {args.model_path}")
        return
    detector = ObjectDetector(args.model_path, config)
    if os.path.isfile(args.input):
        detector.process_single_image(
            args.input, 
            args.confidence, 
            args.output_dir
        )
    elif os.path.isdir(args.input):
        results = detector.process_directory(
            args.input, 
            args.confidence, 
            args.output_dir
        )
        report_path = os.path.join(args.output_dir, 'summary_report.json')
        detector.generate_summary_report(results, report_path)
    else:
        print(f"Входной путь не найден: {args.input}")
if __name__ == "__main__":
    main() 