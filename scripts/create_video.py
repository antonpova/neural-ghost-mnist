import cv2
import os

# ==========================================
# НАСТРОЙКИ
# ==========================================

# 1. Путь к папке с картинками (КОПИРУЙ СВОЙ ПУТЬ СЮДА)
# Внимание: в Python слэши лучше использовать прямые '/' или двойные обратные '\\'
image_folder = '../experiments/run_20251122_224819/animation_hd'

# 2. Имя выходного видео
video_name = 'evolution_timelapse.mp4'

# 3. Количество кадров в секунду (FPS)
# Если картинок 100, то при FPS=10 видео будет длиться 10 секунд
fps = 10

# ==========================================

def make_video():
    # Проверка пути
    if not os.path.exists(image_folder):
        print(f"Ошибка: Папка не найдена -> {image_folder}")
        return

    # Получаем список файлов
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
    # Сортируем (очень важно, иначе кадры будут скакать)
    # Так как мы называли их frame_0001, frame_0002, обычная сортировка сработает
    images.sort()

    if not images:
        print("В папке нет PNG картинок!")
        return

    print(f"Найдено кадров: {len(images)}")

    # Читаем первый кадр, чтобы узнать размеры видео
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Настраиваем кодек (mp4v - стандарт для mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Создаем объект записи (сохраним видео прямо в папку эксперимента)
    output_path = os.path.join(image_folder, video_name)
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Рендеринг видео...")
    
    count = 0
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        
        count += 1
        if count % 10 == 0:
            print(f"Обработано: {count}/{len(images)}")

    video.release()
    cv2.destroyAllWindows()
    
    print("-" * 30)
    print(f"ГОТОВО! Видео сохранено здесь:\n{output_path}")
    print("-" * 30)

if __name__ == "__main__":
    make_video()