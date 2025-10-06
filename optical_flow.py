import cv2
import numpy as np


class ArduPilotOpticalFlow:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.prev_gray = None
        self.flow = None
        self.vx = 0
        self.vy = 0
        self.flow_scale = 3
        self.flow_step = 16

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Конвертируем в grayscale для оптического потока
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Вычисляем оптический поток если есть предыдущий кадр
        if self.prev_gray is not None:
            # Метод Лукаса-Канаде (похож на то, что используется в ArduPilot)
            self.flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Вычисляем средние скорости VX, VY
            self.calculate_velocities()

            # Рисуем векторы оптического потока
            frame = self.draw_flow_vectors(frame)

            # Добавляем информацию о скоростях
            frame = self.draw_velocity_info(frame)

        self.prev_gray = gray.copy()
        return frame

    def calculate_velocities(self):
        """Вычисление скоростей VX, VY как в ArduPilot"""
        if self.flow is None:
            self.vx = 0
            self.vy = 0
            return

        # Берем центральную область для вычисления скоростей
        h, w = self.flow.shape[:2]
        center_h, center_w = h // 2, w // 2
        size = min(h, w) // 3

        # Вычисляем средние скорости в центральной области
        y_start = center_h - size
        y_end = center_h + size
        x_start = center_w - size
        x_end = center_w + size

        # Обрезаем до допустимых границ
        y_start = max(0, y_start)
        y_end = min(h, y_end)
        x_start = max(0, x_start)
        x_end = min(w, x_end)

        # Вычисляем средние скорости
        flow_region = self.flow[y_start:y_end, x_start:x_end]
        self.vx = np.mean(flow_region[..., 0])
        self.vy = np.mean(flow_region[..., 1])

    def draw_flow_vectors(self, frame):
        """Рисует векторы оптического потока"""
        h, w = frame.shape[:2]

        # Рисуем сетку векторов
        for y in range(0, h, self.flow_step):
            for x in range(0, w, self.flow_step):
                fx, fy = self.flow[y, x]

                # Рисуем только значительные движения
                if abs(fx) > 0.5 or abs(fy) > 0.5:
                    end_x = int(x + fx * self.flow_scale)
                    end_y = int(y + fy * self.flow_scale)

                    # Рисуем линию вектора
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y),
                                    (0, 255, 0), 2, tipLength=0.3)

                    # Точка в начале вектора
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        return frame

    def draw_velocity_info(self, frame):
        """Отображает информацию о скоростях VX, VY"""
        # Фон для текста
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)

        # Заголовок
        cv2.putText(frame, "OPTICAL FLOW", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Скорость VX (горизонтальная)
        vx_color = (0, 255, 0) if abs(self.vx) > 0.5 else (100, 100, 100)
        cv2.putText(frame, f"VX: {self.vx:+.2f} px/frame", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vx_color, 2)

        # Скорость VY (вертикальная)
        vy_color = (0, 255, 0) if abs(self.vy) > 0.5 else (100, 100, 100)
        cv2.putText(frame, f"VY: {self.vy:+.2f} px/frame", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vy_color, 2)

        # Общая скорость
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        cv2.putText(frame, f"speed: {speed:.2f} px/frame", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Рисуем центральную область измерения
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        size = min(h, w) // 6
        cv2.rectangle(frame,
                      (center[0] - size, center[1] - size),
                      (center[0] + size, center[1] + size),
                      (255, 0, 0), 2)

        return frame

    def run(self):

        while True:
            frame = self.process_frame()

            if frame is None:
                break

            # Показываем результат
            cv2.imshow('Optical Flow', frame)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Сброс скоростей
                self.vx = 0
                self.vy = 0
                self.prev_gray = None
                print("Скорости сброшены")
            elif key == ord('+'):
                self.flow_scale += 0.5
                print(f"Масштаб векторов: {self.flow_scale}")
            elif key == ord('-'):
                self.flow_scale = max(0.5, self.flow_scale - 0.5)
                print(f"Масштаб векторов: {self.flow_scale}")

        self.cap.release()
        cv2.destroyAllWindows()


# Запуск приложения
if __name__ == "__main__":
    flow = ArduPilotOpticalFlow()
    flow.run()