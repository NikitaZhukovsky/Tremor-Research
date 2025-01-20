import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_tremor(video_path, threshold=30, min_movement_frames=5, pixels_per_mm=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видео.")
        return False

    frame_diffs = []
    amplitudes = []  # Список для амплитуд
    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр.")
        return False

    prev_frame = cv2.resize(prev_frame, (640, 480))
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    total_frames = 0
    movement_frames = 0

    # Получение частоты кадров
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_frame_gray, frame_gray)
        diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        frame_diffs.append(diff)

        movement = np.sum(diff)
        if movement > 1000:
            movement_frames += 1
            amplitudes.append(movement)  # Сохраняем амплитуду

        total_frames += 1
        prev_frame_gray = frame_gray

    cap.release()

    if len(frame_diffs) == 0:
        print("Нет доступных кадров для анализа.")
        return False

    data = np.array([d.flatten() for d in frame_diffs], dtype=np.float32)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(data)

    labels = kmeans.labels_
    tremor_detected = np.sum(labels == 1) > len(labels) / 2

    max_amplitude = max(amplitudes) if amplitudes else 0

    if total_frames > 0 and fps > 0:
        frequency = (movement_frames / (total_frames / fps))
    else:
        frequency = 0

    print("Общее количество обработанных кадров:", total_frames)
    print("Количество кадров с заметными изменениями:", movement_frames)
    print("Порог для определения изменений:", threshold)
    print("Обнаружен тремор:", tremor_detected)
    print("Максимальная амплитуда движения:", max_amplitude)
    print("Частота тремора (изменений/сек):", frequency)

    movement_percentage = (movement_frames / total_frames) * 100 if total_frames > 0 else 0
    print("Процент кадров с тремором:", movement_percentage)

    return tremor_detected


video_path = 'video2.mp4'
if not os.path.isfile(video_path):
    print("Файл не найден:", video_path)
else:
    tremor = detect_tremor(video_path)