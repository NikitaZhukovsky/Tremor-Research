import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_tremor(video_path, threshold=10, min_movement_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видео.")
        return False

    amplitudes = [0]
    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр.")
        return False

    prev_frame = cv2.resize(prev_frame, (640, 480))
    previous_area = 0

    total_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    output_width, output_height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (output_width, output_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([130, 50, 50])
        upper_color = np.array([170, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            if w >= 15 and h >= 15:
                area = cv2.contourArea(max_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


                if previous_area > 0:
                    amplitude_in_mm = (area - previous_area) / (w / 15.0)
                    if amplitude_in_mm > 0:
                        amplitudes.append(amplitude_in_mm)

                previous_area = area

        total_frames += 1

        resized_frame = cv2.resize(frame, (output_width, output_height))
        out.write(resized_frame)

        cv2.imshow('Tremor Detection', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if amplitudes:
        time_values = np.arange(len(amplitudes)) / fps
        plt.figure(figsize=(10, 5))
        plt.plot(time_values, amplitudes, label='Амплитуда движения (мм)', color='blue')
        plt.title('Изменение амплитуды движения по времени')
        plt.xlabel('Время (секунды)')
        plt.ylabel('Амплитуда (мм)')
        plt.ylim(0, 30)
        plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
        plt.xlim(0, time_values[-1])
        plt.grid()
        plt.legend()
        plt.show()

    return True

video_path = 'video6.mp4'
if not os.path.isfile(video_path):
    print("Файл не найден:", video_path)
else:
    tremor = detect_tremor(video_path)