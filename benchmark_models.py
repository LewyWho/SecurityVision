import os
import torch
import time
import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from models import get_model
from torchvision import transforms
class ModelBenchmark:
    def __init__(self, image_size=224, num_classes=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_input = torch.randn(1, 3, image_size, image_size).to(self.device)
    def benchmark_model(self, model_name, num_runs=100):
        print(f"\nТестирование модели: {model_name}")
        try:
            model = get_model(model_name, num_classes=self.num_classes, input_size=512)
            model.to(self.device)
            model.eval()
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 * 1024)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(self.test_input)
            inference_times = []
            with torch.no_grad():
                for _ in tqdm(range(num_runs), desc=f"Инференс {model_name}"):
                    start_time = time.time()
                    _ = model(self.test_input)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            fps = 1.0 / avg_time
            return {
                'model_name': model_name,
                'success': True,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': model_size_mb,
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'fps': fps,
                'device': str(self.device)
            }
        except Exception as e:
            print(f"Ошибка при тестировании {model_name}: {e}")
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    def run_full_benchmark(self, models_to_test=None):
        if models_to_test is None:
            models_to_test = [
                'yolo',
                'ssd',
                'efficientdet',
            ]
        print("ЗАПУСК БЕНЧМАРКА МОДЕЛЕЙ")
        print("=" * 60)
        print(f"Устройство: {self.device}")
        print(f"Размер изображения: {self.image_size}x{self.image_size}")
        print(f"Количество тестов: 100 на модель")
        results = []
        for model_name in models_to_test:
            result = self.benchmark_model(model_name)
            results.append(result)
            if result['success']:
                print(f"{model_name}: {result['fps']:.1f} FPS, {result['model_size_mb']:.1f} MB")
            else:
                print(f"{model_name}: Ошибка")
        successful_results = [r for r in results if r['success']]
        successful_results.sort(key=lambda x: x['fps'], reverse=True)
        print(f"\nРЕЗУЛЬТАТЫ БЕНЧМАРКА")
        print("=" * 60)
        print(f"{'Модель':<15} {'FPS':<8} {'Время (мс)':<12} {'Размер (MB)':<12} {'Параметры':<12}")
        print("-" * 60)
        for result in successful_results:
            print(f"{result['model_name']:<15} "
                  f"{result['fps']:<8.1f} "
                  f"{result['avg_inference_time']*1000:<12.2f} "
                  f"{result['model_size_mb']:<12.1f} "
                  f"{result['total_params']/1e6:<12.1f}M")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_benchmark_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'benchmark_info': {
                    'timestamp': timestamp,
                    'device': str(self.device),
                    'image_size': self.image_size,
                    'num_runs': 100
                },
                'results': results,
                'ranking': [r['model_name'] for r in successful_results]
            }, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты сохранены в: {output_file}")
        print(f"\nРЕКОМЕНДАЦИИ:")
        if successful_results:
            fastest = successful_results[0]
            print(f"Самая быстрая модель: {fastest['model_name']} ({fastest['fps']:.1f} FPS)")
            if fastest['fps'] > 30:
                print("Отлично! Модель подходит для real-time приложений")
            elif fastest['fps'] > 10:
                print("Хорошо! Модель подходит для быстрой обработки")
            else:
                print("Медленно! Рассмотрите более легкие модели")
        return results
def main():
    parser = argparse.ArgumentParser(description='Бенчмарк скорости моделей')
    parser.add_argument('--models', nargs='+', default=['yolo', 'ssd', 'efficientdet'], help='Модели для тестирования')
    parser.add_argument('--image_size', type=int, default=224, help='Размер изображения')
    parser.add_argument('--num_runs', type=int, default=100, help='Количество тестов')
    args = parser.parse_args()
    benchmark = ModelBenchmark(image_size=args.image_size)
    results = benchmark.run_full_benchmark(args.models)
    return results
if __name__ == "__main__":
    main() 