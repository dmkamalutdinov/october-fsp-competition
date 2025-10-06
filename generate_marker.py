import cv2
import numpy as np
import argparse


def generate_marker():
    parser = argparse.ArgumentParser(description='Словарь 4x4_1000')
    parser.add_argument('--marker_id', type=int, default=0, help='ID маркера (0-999)')
    parser.add_argument('--output', type=str, default='aruco_marker.png', help='название файла')
    parser.add_argument('--size', type=int, default=500, help='размер маркера в пикселях')

    args = parser.parse_args()

    # проверка корректности
    if args.marker_id < 0 or args.marker_id > 999:
        print("id маркера должен быть между 0 и 999")
        return

    # загрузка словаря
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

    # генерация маркера
    marker_image = np.zeros((args.size, args.size), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, args.marker_id, args.size, marker_image, 1)

    cv2.imwrite(args.output, marker_image)
    cv2.imshow(f'ID {args.marker_id}', marker_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_marker()