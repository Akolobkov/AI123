
import cv2
import numpy as np
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Класс для хранения результата сопоставления"""
    success: bool = False
    homography: Optional[np.ndarray] = None
    good_matches: List[cv2.DMatch] = None
    keypoints_template: List[cv2.KeyPoint] = None
    keypoints_scene: List[cv2.KeyPoint] = None
    execution_time: float = 0.0
    message: str = ""
    match_count: int = 0


class SIFTMatcher:

    def __init__(
            self,
            nfeatures: int = 0,
            nOctaveLayers: int = 3,
            contrastThreshold: float = 0.04,
            edgeThreshold: int = 10,
            sigma: float = 1.6,
            uniqueness_threshold: float = 1,
            ransac_reprojection_threshold: float = 2.0,
            knn_k: int = 2
    ):
        """
        Инициализация SIFT детектора

        Args:
            nfeatures: Количество лучших признаков (0 = все)
            nOctaveLayers: Количество слоев в октаве
            contrastThreshold: Порог контрастности
            edgeThreshold: Порог для фильтрации краев
            sigma: Сигма для начального размытия
            uniqueness_threshold: Порог уникальности (Lowe's ratio test)
            ransac_reprojection_threshold: Порог для RANSAC
            knn_k: Количество ближайших соседей
        """
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        # Параметры сопоставления
        self.uniqueness_threshold = uniqueness_threshold
        self.ransac_reprojection_threshold = ransac_reprojection_threshold
        self.knn_k = knn_k

        # Создаем SIFT детектор
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )

        # Создаем BFMatcher (Brute-Force)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Хранилище для шаблона
        self.template_gray = None
        self.template_color = None
        self.template_keypoints = None
        self.template_descriptors = None
        self.template_path = None

        # Для измерения времени
        self.last_processing_time = 0.0

    def load_template(self, image_path: str) -> bool:
        """
        Загрузка шаблона из файла

        Args:
            image_path: Путь к файлу

        Returns:
            bool: Успешность загрузки
        """
        try:
            self.template_color = cv2.imread(image_path)
            if self.template_color is None:
                raise ValueError(f"Не удалось загрузить: {image_path}")

            self.template_gray = cv2.cvtColor(self.template_color, cv2.COLOR_BGR2GRAY)

            # Находим ключевые точки и дескрипторы
            self.template_keypoints, self.template_descriptors = (
                self.sift.detectAndCompute(self.template_gray, None)
            )

            if self.template_keypoints is None or len(self.template_keypoints) < 4:
                raise ValueError("Слишком мало ключевых точек")

            self.template_path = image_path
            print(f"✅ Шаблон загружен: {len(self.template_keypoints)} ключевых точек")
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки шаблона: {e}")
            return False

    def load_template_from_array(self, image: np.ndarray) -> bool:
        """
        Загрузка шаблона из массива numpy

        Args:
            image: Изображение в формате BGR

        Returns:
            bool: Успешность загрузки
        """
        try:
            self.template_color = image.copy()
            self.template_gray = cv2.cvtColor(self.template_color, cv2.COLOR_BGR2GRAY)

            self.template_keypoints, self.template_descriptors = (
                self.sift.detectAndCompute(self.template_gray, None)
            )

            if self.template_keypoints is None or len(self.template_keypoints) < 4:
                raise ValueError("Слишком мало ключевых точек")

            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки шаблона: {e}")
            return False

    def find_object(
            self,
            scene_image: np.ndarray,
            draw_keypoints: bool = False,
            draw_matches: bool = False,
            min_match_count: int = 4
    ) -> Tuple[MatchResult, Optional[np.ndarray]]:
        """
        Поиск объекта на сцене

        Args:
            scene_image: Изображение сцены (BGR)
            draw_keypoints: Рисовать ли ключевые точки
            draw_matches: Рисовать ли линии совпадений
            min_match_count: Минимальное число совпадений для гомографии

        Returns:
            MatchResult: Результат сопоставления
            output_image: Изображение с результатами
        """
        start_time = time.time()
        result = MatchResult()

        # Проверка наличия шаблона
        if self.template_gray is None or self.template_descriptors is None:
            result.message = "Шаблон не загружен"
            return result, None

        if scene_image is None or scene_image.size == 0:
            result.message = "Сцена пустая"
            return result, None

        try:
            # 1. Преобразование сцены в оттенки серого
            scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

            # 2. Нахождение ключевых точек на сцене
            scene_keypoints, scene_descriptors = self.sift.detectAndCompute(
                scene_gray, None
            )

            if scene_keypoints is None or len(scene_keypoints) < 4:
                result.message = "На сцене мало ключевых точек"
                return result, scene_image

            # 3. Сопоставление дескрипторов (KNN Match)
            matches = self.bf_matcher.knnMatch(
                self.template_descriptors,
                scene_descriptors,
                k=self.knn_k
            )

            # 4. Применение Lowe's ratio test (аналог VoteForUniqueness)
            good_matches = []
            for m, n in matches:
                if m.distance < self.uniqueness_threshold * n.distance:
                    good_matches.append(m)

            result.good_matches = good_matches
            result.match_count = len(good_matches)

            # 5. Проверка количества совпадений
            if len(good_matches) < min_match_count:
                result.message = f"Недостаточно совпадений: {len(good_matches)}"
                output_img = self._draw_result(
                    scene_image,
                    scene_keypoints,
                    good_matches,
                    draw_keypoints,
                    draw_matches,
                    template_img=self.template_color,
                    template_kp=self.template_keypoints
                )
                result.execution_time = time.time() - start_time
                return result, output_img

            # 6. Подготовка точек для гомографии
            src_pts = np.float32([
                self.template_keypoints[m.queryIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)

            dst_pts = np.float32([
                scene_keypoints[m.trainIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)

            # 7. Вычисление гомографии с помощью RANSAC
            homography, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.ransac_reprojection_threshold
            )

            result.homography = homography
            result.keypoints_template = self.template_keypoints
            result.keypoints_scene = scene_keypoints

            # 8. Отрисовка результата
            output_img = self._draw_result(
                scene_image,
                scene_keypoints,
                good_matches,
                draw_keypoints,
                draw_matches,
                homography=homography,
                template_img=self.template_color,
                template_kp=self.template_keypoints,
                mask=mask
            )

            # 9. Определение успеха
            if homography is not None:
                result.success = True
                result.message = "✅ Объект найден!"
            else:
                result.message = "❌ Объект не найден"

            result.execution_time = time.time() - start_time
            return result, output_img

        except Exception as e:
            result.message = f"Ошибка: {e}"
            return result, scene_image

    def _draw_result(
            self,
            scene_img: np.ndarray,
            scene_kp: List[cv2.KeyPoint],
            good_matches: List[cv2.DMatch],
            draw_keypoints: bool,
            draw_matches: bool,
            homography: Optional[np.ndarray] = None,
            template_img: Optional[np.ndarray] = None,
            template_kp: Optional[List[cv2.KeyPoint]] = None,
            mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Отрисовка результатов на изображении
        """
        output = scene_img.copy()

        # Рисуем ключевые точки (синим, как в методичке)
        if draw_keypoints and scene_kp:
            cv2.drawKeypoints(
                output,
                scene_kp,
                output,
                color=(255, 0, 0),  # Синий
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        # Рисуем линии совпадений, если включено
        if draw_matches and template_img is not None and template_kp is not None and scene_kp is not None:
            if good_matches:
                # Фильтруем совпадения, проверяя валидность индексов
                valid_matches = []
                for match in good_matches:
                    if (match.trainIdx < len(template_kp) and
                            match.queryIdx < len(scene_kp)):
                        valid_matches.append(match)

                if valid_matches:
                    # Конвертируем в цветные, если нужно
                    if len(output.shape) == 2:
                        output_color = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                    else:
                        output_color = output.copy()

                    if len(template_img.shape) == 2:
                        template_color = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
                    else:
                        template_color = template_img.copy()

                    match_img = cv2.drawMatches(
                        output_color, scene_kp,
                        template_color, template_kp,
                        valid_matches[:50], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                        matchesThickness=1
                    )
                    output = match_img
                else:
                    print("No valid matches after filtering")
        else:
            # Просто склеиваем изображения
            if template_img is not None:
                # Приводим к одинаковой высоте
                h1, w1 = template_img.shape[:2]
                h2, w2 = output.shape[:2]
                if h1 != h2:
                    scale = h2 / h1
                    new_w = int(w1 * scale)
                    template_resized = cv2.resize(template_img, (new_w, h2))
                else:
                    template_resized = template_img.copy()

                # Добавляем красную рамку вокруг шаблона (как в методичке)
                cv2.rectangle(
                    template_resized,
                    (1, 1),
                    (template_resized.shape[1] - 3, template_resized.shape[0] - 3),
                    (0, 0, 255),  # Красный
                    2
                )

                # Рисуем ключевые точки на шаблоне
                if draw_keypoints and template_kp:
                    cv2.drawKeypoints(
                        template_resized,
                        template_kp,
                        template_resized,
                        color=(255, 0, 0),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )

                # Объединяем горизонтально
                output = np.hstack([output, template_resized])

        # Рисуем прямоугольник вокруг найденного объекта (красный)
        if homography is not None and self.template_gray is not None:
            h, w = self.template_gray.shape[:2]

            # Углы шаблона
            pts = np.float32([
                [0, 0],
                [0, h - 1],
                [w - 1, h - 1],
                [w - 1, 0]
            ]).reshape(-1, 1, 2)

            # Проецируем на сцену
            dst = cv2.perspectiveTransform(pts, homography)

            # Рисуем прямоугольник
            cv2.polylines(
                output,
                [np.int32(dst)],
                True,
                (0, 0, 255),  # Красный
                2
            )

        return output

    def get_template_info(self) -> dict:
        """Получение информации о шаблоне"""
        if self.template_gray is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "path": self.template_path,
            "width": self.template_gray.shape[1],
            "height": self.template_gray.shape[0],
            "keypoints": len(self.template_keypoints) if self.template_keypoints else 0,
            "algorithm": "SIFT"
        }

    def set_parameters(self, **kwargs):
        """
        Обновление параметров
        """
        if 'contrastThreshold' in kwargs:
            self.contrastThreshold = kwargs['contrastThreshold']
        if 'edgeThreshold' in kwargs:
            self.edgeThreshold = kwargs['edgeThreshold']
        if 'uniqueness_threshold' in kwargs:
            self.uniqueness_threshold = kwargs['uniqueness_threshold']
        if 'ransac_reprojection_threshold' in kwargs:
            self.ransac_reprojection_threshold = kwargs['ransac_reprojection_threshold']

        # Пересоздаем SIFT с новыми параметрами
        self.sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.nOctaveLayers,
            contrastThreshold=self.contrastThreshold,
            edgeThreshold=self.edgeThreshold,
            sigma=self.sigma
        )

        # Пересчитываем дескрипторы шаблона
        if self.template_gray is not None:
            self.template_keypoints, self.template_descriptors = (
                self.sift.detectAndCompute(self.template_gray, None)
            )