import sys
import subprocess

def check_version(package_name):
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        return "Не установлен"

def main():
    print("ПРОВЕРКА СОВМЕСТИМОСТИ ВЕРСИЙ")
    print("=" * 50)
    
    packages = [
        'torch',
        'torchvision', 
        'torchmetrics',
        'opencv-python',
        'albumentations',
        'timm'
    ]
    
    for package in packages:
        version = check_version(package)
        print(f"{package:20} : {version}")
    
    print("\n" + "=" * 50)
    
    try:
        import torchvision
        print(f"Torchvision {torchvision.__version__} установлен")
        
    except ImportError as e:
        print(f"Ошибка импорта torchvision: {e}")

if __name__ == "__main__":
    main() 