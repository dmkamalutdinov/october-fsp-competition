import cv2
import numpy as np
import math
import time


class OpticalFlowPX4:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.prev_frame = None
        self.prev_time = None
        self.hann_window = None
        self.roi_size = 128
        self.roi_rad = 0.0
        self.roi = None
        self.flow_x = 0.0
        self.flow_y = 0.0
        self.quality = 0.0
        self.integration_time = 0.0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def setup_camera(self, width=640, height=480):
        self.camera_matrix = np.array([
            [width, 0, width / 2],
            [0, height, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

    def calculate_roi(self, frame):
        """Вычисление ROI области интереса как в C++ коде"""
        h, w = frame.shape[:2]

        if self.roi_rad != 0:
            # Упрощенная версия ROI по углам (как в C++)
            roi_size = int(min(h, w) * 0.7)  # Примерный размер
            self.roi = (w // 2 - roi_size // 2, h // 2 - roi_size // 2, roi_size, roi_size)
            print(f"ROI: {self.roi[0]} {self.roi[1]} - {self.roi[0] + self.roi[2]} {self.roi[1] + self.roi[3]}")
        else:
            self.roi = (w // 2 - self.roi_size // 2, h // 2 - self.roi_size // 2,
                        self.roi_size, self.roi_size)

    def process_frame(self, frame):
        """Основная обработка кадра для оптического потока"""
        current_time = time.time()

        # Конвертируем в grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Применяем ROI если задан
        if self.roi is not None:
            x, y, w, h = self.roi
            processing_frame = gray[y:y + h, x:x + w]
        else:
            processing_frame = gray

        # Конвертируем в float для точных вычислений
        gray_float = processing_frame.astype(np.float32)

        if self.prev_frame is None:
            # Первый кадр - инициализация
            self.prev_frame = gray_float
            self.prev_time = current_time
            self.hann_window = cv2.createHanningWindow((gray_float.shape[1], gray_float.shape[0]), cv2.CV_32F)
            return None

        # Проверяем время между кадрами
        time_diff = current_time - self.prev_time
        if time_diff > 0.1:  # Слишком большая задержка
            self.prev_frame = gray_float
            self.prev_time = current_time
            return None

        try:
            # Вычисляем фазовую корреляцию (как в оригинальном C++ коде)
            shift, response = cv2.phaseCorrelate(self.prev_frame, gray_float, self.hann_window)
        except:
            # В случае ошибки сбрасываем состояние
            self.prev_frame = gray_float
            self.prev_time = current_time
            return None

        # Сохраняем смещение в пикселях
        pixel_shift_x, pixel_shift_y = shift

        # Публикуем неискаженный поток
        flow_x_rad, flow_y_rad = self.undistort_flow(pixel_shift_x, pixel_shift_y, processing_frame.shape)

        # Преобразуем в систему координат FCU (как в PX4)
        flow_fcu_x = flow_y_rad  # +y означает вращение против часовой стрелки вокруг Y
        flow_fcu_y = -flow_x_rad  # +x означает вращение по часовой стрелке вокруг X

        # Сохраняем результаты
        self.flow_x = flow_fcu_x
        self.flow_y = flow_fcu_y
        self.quality = response
        self.integration_time = time_diff

        # Обновляем предыдущий кадр
        self.prev_frame = gray_float
        self.prev_time = current_time

        # Вычисляем FPS
        self.frame_count += 1
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

        return flow_fcu_x, flow_fcu_y, response, pixel_shift_x, pixel_shift_y

    def undistort_flow(self, pixel_x, pixel_y, frame_shape):
        """Коррекция дисторсии для потока как в C++ коде"""
        h, w = frame_shape

        # Центр потока
        flow_center_x = w / 2
        flow_center_y = h / 2

        # Смещаем координаты (как в C++ коде)
        shift_x = pixel_x + flow_center_x
        shift_y = pixel_y + flow_center_y

        # Точки для коррекции дисторсии
        points_dist = np.array([[[shift_x, shift_y]]], dtype=np.float32)

        if self.camera_matrix is not None and self.dist_coeffs is not None:
            # Коррекция дисторсии (как в C++ cv::undistortPoints)
            points_undist = cv2.undistortPoints(points_dist, self.camera_matrix,
                                                self.dist_coeffs, P=self.camera_matrix)
            point_undist = points_undist[0][0]
        else:
            point_undist = points_dist[0][0]

        # Возвращаем к центру (как в C++ коде)
        point_undist[0] -= flow_center_x
        point_undist[1] -= flow_center_y

        # Конвертируем в радианы (как в C++ коде)
        focal_length_x = self.camera_matrix[0, 0] if self.camera_matrix is not None else w
        focal_length_y = self.camera_matrix[1, 1] if self.camera_matrix is not None else h

        flow_x_rad = math.atan2(point_undist[0], focal_length_x)
        flow_y_rad = math.atan2(point_undist[1], focal_length_y)

        return flow_x_rad, flow_y_rad

    def draw_flow(self, frame, x, y, quality):
        """
        Точная копия C++ функции drawFlow
        Рисует круг и линию, указывающую направление смещения
        """
        brightness = int((1 - quality) * 25)
        color = (brightness, brightness, brightness)
        radius = math.sqrt(x * x + y * y)

        # Центр кадра
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        center = (center_x, center_y)

        # Рисуем круг и линию как в C++ коде
        cv2.circle(frame, center, int(radius * 5), color, 3, cv2.LINE_AA)
        cv2.line(frame, center,
                 (center_x + int(x * 5), center_y + int(y * 5)),
                 color, 3, cv2.LINE_AA)

        return frame

    def draw_debug_info(self, frame, pixel_shift_x, pixel_shift_y, quality):
        """Отрисовка отладочной информации в стиле C++ кода"""
        # Создаем монохромную версию для отрисовки потока
        if len(frame.shape) == 3:
            debug_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        else:
            debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Применяем ROI визуализацию
        if self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Рисуем поток в стиле C++ кода
        debug_frame = self.draw_flow(debug_frame, pixel_shift_x, pixel_shift_y, quality)

        # Добавляем текстовую информацию (аналогично C++)
        info_text = [
            f"PX4 Optical Flow",
            f"Integrated X: {self.flow_x:.6f} rad",
            f"Integrated Y: {self.flow_y:.6f} rad",
            f"Quality: {int(quality * 255)}/255",
            f"Integration: {self.integration_time * 1e6:.0f} us",
            f"FPS: {self.fps}",
            f"Pixel Shift: ({pixel_shift_x:.2f}, {pixel_shift_y:.2f})"
        ]

        # Фон для текста (черный прямоугольник как в C++)
        cv2.rectangle(debug_frame, (10, 10), (450, 160), (0, 0, 0), -1)

        for i, text in enumerate(info_text):
            y_pos = 35 + i * 20
            color = (255, 255, 255)  # Белый текст как в C++

            # Первая строка - заголовок
            if i == 0:
                cv2.putText(debug_frame, text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(debug_frame, text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return debug_frame


def main():
    # Инициализация оптического потока
    optical_flow = OpticalFlowPX4()

    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Настройка камеры
    optical_flow.setup_camera(640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Первый кадр - вычисляем ROI
        if optical_flow.roi is None:
            optical_flow.calculate_roi(frame)

        # Обработка оптического потока
        result = optical_flow.process_frame(frame)

        if result is not None:
            flow_fcu_x, flow_fcu_y, quality, pixel_shift_x, pixel_shift_y = result

            # Отрисовка в стиле C++ кода
            debug_frame = optical_flow.draw_debug_info(frame, pixel_shift_x, pixel_shift_y, quality)
        else:
            # Если нет результата, показываем исходный кадр
            debug_frame = frame.copy()
            cv2.putText(debug_frame, "Initializing...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Показ результата
        cv2.imshow('PX4 Optical Flow', debug_frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Сброс оптического потока
            optical_flow.prev_frame = None
            print("Optical flow reset")
        elif key == ord('c'):
            # Перекалибровка ROI
            optical_flow.roi = None
            optical_flow.calculate_roi(frame)
            print("ROI recalibrated")
        elif key == ord('1'):
            # Уменьшение ROI
            optical_flow.roi_size = max(64, optical_flow.roi_size - 16)
            optical_flow.roi = None
            print(f"ROI size: {optical_flow.roi_size}")
        elif key == ord('2'):
            # Увеличение ROI
            optical_flow.roi_size = min(256, optical_flow.roi_size + 16)
            optical_flow.roi = None
            print(f"ROI size: {optical_flow.roi_size}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()