import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_command(command, description):
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
        print(f"\n❌ Ошибка при выполнении {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Быстрый запуск обучения моделей')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', 'full', 'compare'],
                       help='Режим обучения')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU (если доступен)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Количество эпох (переопределяет режим)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Размер батча (переопределяет режим)')
    
    args = parser.parse_args()
    
    configs = {
        'quick': {
            'epochs': 10,
            'batch_size': 8,
            'models': ['yolo', 'efficientdet'],
            'description': 'Быстрое обучение (10 эпох, 2 модели)'
        },
        'full': {
            'epochs': 50,
            'batch_size': 16,
            'models': ['all'],
            'description': 'Полное обучение (50 эпох, все модели)'
        },
        'compare': {
            'epochs': 30,
            'batch_size': 12,
            'models': ['all'],
            'description': 'Сравнительное обучение (30 эпох, все модели)'
        }
    }
    
    config = configs[args.mode]
    
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    print(f"🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ")
    print(f"Режим: {args.mode}")
    print(f"Описание: {config['description']}")
    print(f"Эпохи: {config['epochs']}")
    print(f"Размер батча: {config['batch_size']}")
    print(f"Модели: {config['models']}")
    print(f"GPU: {'Да' if args.gpu else 'Нет'}")
    
    if not os.path.exists('Data/train.json') or not os.path.exists('Data/test.json'):
        print("❌ Ошибка: Файлы данных не найдены!")
        print("Убедитесь, что файлы Data/train.json и Data/test.json существуют")
        return False
    
    print("\n🔍 Проверка зависимостей...")
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import tqdm
        print("✅ Все зависимости установлены")
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    os.makedirs('inference_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model in config['models']:
        if model == 'all':
            command = f"python train.py --model all --epochs {config['epochs']} --batch_size {config['batch_size']} --lr 0.001"
        else:
            command = f"python train.py --model {model} --epochs {config['epochs']} --batch_size {config['batch_size']} --lr 0.001"
        
        success = run_command(command, f"Обучение модели {model}")
        
        if not success:
            print(f"❌ Обучение модели {model} завершилось с ошибкой")
            continue
    
    print("\n📊 Запуск оценки моделей...")
    eval_command = "python evaluate.py --evaluate_all --batch_size 16"
    eval_success = run_command(eval_command, "Оценка всех моделей")
    
    if eval_success:
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("\n📁 Результаты сохранены в:")
        print("  - models/ - обученные модели")
        print("  - evaluation_results/ - результаты оценки")
        print("  - model_comparison.json - сравнение моделей")
        
        if os.path.exists('model_comparison.json'):
            import json
            with open('model_comparison.json', 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if results:
                best_model = max(results, key=lambda x: x['accuracy'])
                print(f"\n🏆 Лучшая модель: {best_model['model_name']} "
                      f"(точность: {best_model['accuracy']*100:.2f}%)")
        
        print(f"\n💡 Для инференса используйте:")
        print(f"python inference.py --model_path models/[model_name]/best.pth --input [path_to_images]")
        
    else:
        print("\n⚠️ Обучение завершено, но оценка не удалась")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 