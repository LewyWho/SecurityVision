import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

def run_command_with_fallback(command, description, model_name):
    print(f"\n{'='*60}")
    print(f"ВЫПОЛНЕНИЕ: {description}")
    print(f"{'='*60}")
    print(f"Команда: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"\n✅ {description} завершено успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Ошибка при выполнении {description}: {e}")
        
        if "faster_rcnn" in model_name or "retinanet" in model_name:
            print(f"🔄 Попытка с упрощенной версией {model_name}...")
            
            simplified_command = command.replace(
                f"--model {model_name}", 
                f"--model {model_name} --use_simple"
            )
            
            try:
                result = subprocess.run(simplified_command, shell=True, check=True,
                                      capture_output=False, text=True)
                print(f"\n✅ {description} завершено с упрощенной моделью!")
                return True
            except subprocess.CalledProcessError as e2:
                print(f"\n❌ Ошибка даже с упрощенной моделью: {e2}")
                return False
        else:
            return False

def check_environment():
    print("ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 40)
    
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import tqdm
        import albumentations
        import timm
        
        print(f"PyTorch: {torch.__version__}")
        print(f"Torchvision: {torchvision.__version__}")
        print(f"NumPy: {np.__version__}")
        print(f"Albumentations: {albumentations.__version__}")
        print(f"TIMM: {timm.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA доступен: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA недоступен, используется CPU")
        
        return True
    except ImportError as e:
        print(f"Отсутствуют зависимости: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False

def main():
    parser = argparse.ArgumentParser(description='Надежное обучение моделей детекции')
    parser.add_argument('--models', nargs='+', 
                       default=['yolo', 'efficientdet', 'ssd'],
                       help='Модели для обучения')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512, help='Размер изображения')
    parser.add_argument('--check_compatibility', action='store_true',
                       help='Проверить совместимость перед обучением')
    
    args = parser.parse_args()
    
    print("НАДЕЖНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ ДЕТЕКЦИИ")
    print("=" * 60)
    print(f"Модели: {args.models}")
    print(f"Эпохи: {args.epochs}")
    print(f"Размер батча: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Размер изображения: {args.image_size}")
    
    if not check_environment():
        return False
    
    if args.check_compatibility:
        print("\nПРОВЕРКА СОВМЕСТИМОСТИ")
        try:
            subprocess.run(['python', 'check_compatibility.py'], check=True)
        except subprocess.CalledProcessError:
            print("Проблемы с совместимостью обнаружены")
    
    if not os.path.exists('Data/train.json') or not os.path.exists('Data/test.json'):
        print("❌ Ошибка: Файлы данных не найдены!")
        print("Убедитесь, что файлы Data/train.json и Data/test.json существуют")
        return False
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    os.makedirs('inference_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    for model in args.models:
        command = (f"python train.py --model {model} --epochs {args.epochs} "
                  f"--batch_size {args.batch_size} --lr {args.lr} --image_size {args.image_size}")
        
        success = run_command_with_fallback(
            command, 
            f"Обучение модели {model}", 
            model
        )
        
        results.append({
            'model': model,
            'success': success,
            'timestamp': timestamp
        })
        
        if not success:
            print(f"❌ Обучение модели {model} завершилось с ошибкой")
            continue
    
    with open(f'training_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    successful_models = [r for r in results if r['success']]
    print(f"\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print(f"Успешно обучено: {len(successful_models)}/{len(args.models)}")
    
    if successful_models:
        print("\nУспешно обученные модели:")
        for result in successful_models:
            print(f"  - {result['model']}")
        
        print("\nЗАПУСК ОЦЕНКИ УСПЕШНЫХ МОДЕЛЕЙ")
        for result in successful_models:
            model_path = f"models/{result['model']}/best.pth"
            if os.path.exists(model_path):
                eval_command = (f"python evaluate.py --model_path {model_path} "
                              f"--batch_size {args.batch_size} --image_size {args.image_size}")
                
                try:
                    subprocess.run(eval_command, shell=True, check=True)
                    print(f"{result['model']} оценена успешно")
                except subprocess.CalledProcessError:
                    print(f"Ошибка при оценке {result['model']}")
    
    print(f"\nРезультаты сохранены в training_results_{timestamp}.json")
    print(f"Модели сохранены в директории models/")
    
    return len(successful_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 