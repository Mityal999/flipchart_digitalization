import cv2
import numpy as np
import os

class ImageProcessing:
    # Класс для обработки изображений
    def __init__(self, image_path, name):
        self.image_path = image_path
        self.ocr_img = None
        self.h_src, self.w_src, self.c_src = None, None, None
        self.name = name
        self.approx_max = None
        self.im_out = None
        self.canvas = None

    def load_image(self):
        # Загрузка изображения
        self.ocr_img = cv2.imread(self.image_path)
        self.h_src, self.w_src, self.c_src = self.ocr_img.shape

    def resize_image(self, target_long_side, keep_aspect_ratio=True):
        # Масштабирование изображения с сохранением пропорций
        height, width = self.ocr_img.shape[:2]
        if keep_aspect_ratio:
            scale = target_long_side / max(height, width)
            new_dimensions = (int(width * scale), int(height * scale))
        else:
            new_dimensions = (target_long_side, target_long_side)
        self.ocr_img = cv2.resize(self.ocr_img, new_dimensions)
        self.h_src, self.w_src = new_dimensions

    def preprocess_image(self):
        # Предварительная обработка изображения
        ocr_img_gray = cv2.cvtColor(self.ocr_img, cv2.COLOR_BGR2GRAY)
        ocr_img_gray = cv2.GaussianBlur(ocr_img_gray, (3, 3), 1)
        _, self.ocr_img_thresh = cv2.threshold(ocr_img_gray, 200, 255, cv2.THRESH_BINARY)

    def find_contours(self):
        # Поиск контуров
        contours, _ = cv2.findContours(self.ocr_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.ocr_img_contours = contours

    def process_contours(self):
        # Обработка контуров
        if not self.ocr_img_contours:
            return  # Нет контуров для обработки
        largest_contour = max(self.ocr_img_contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        self.draw_img = cv2.drawContours(self.ocr_img.copy(), [approx], -1, (0, 255, 0), 2)
        self.approx_max = approx

    def sort_dot_cnt(self, kps):
        # Сортировка точек контура
        rect = np.zeros((4, 2), dtype='float32')
        s = kps.sum(axis=1)
        diff = np.diff(kps, axis=1)
        rect[0], rect[2] = kps[np.argmin(s)], kps[np.argmax(s)]
        rect[1], rect[3] = kps[np.argmin(diff)], kps[np.argmax(diff)]
        return rect

    def perspective_transform(self, apply_transform=True):
        # Перспективное преобразование
        if not apply_transform or self.approx_max is None or len(self.approx_max) != 4:
            self.im_out = self.ocr_img_thresh
            return
        rect_ordered = self.sort_dot_cnt(self.approx_max.reshape(4, 2))
        pts_src = rect_ordered.astype('float32')
        pts_dst = np.array([[0, 0], [self.w_src, 0], [self.w_src, self.h_src], [0, self.h_src]], dtype='float32')
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        self.im_out = cv2.warpPerspective(self.ocr_img_thresh, M, (self.w_src, self.h_src))

    def create_canvas(self):
        # Создание холста для результата
        canvas = np.full((1800, 1500), 255, dtype=np.uint8)
        h_output, w_output = self.im_out.shape[:2]
        x_offset, y_offset = (1500 - w_output) // 2, (1800 - h_output) // 2
        canvas[y_offset:y_offset + h_output, x_offset:x_offset + w_output] = self.im_out
        self.canvas = canvas

    def save_results(self, canvas_path):
        cv2.imwrite(canvas_path, self.canvas)

    def process_image(self, tmp_folder, target_long_side=1400):
        self.load_image()
        self.resize_image(target_long_side)
        self.preprocess_image()
        self.find_contours()
        self.process_contours()
        self.perspective_transform()
        self.create_canvas()
        canvas_path = f'{tmp_folder}/{self.name}.jpg'
        self.save_results(canvas_path)