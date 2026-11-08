# lab2_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, IntVar, BooleanVar
from PIL import Image, ImageTk
import cv2
import threading
from sift_matcher import SIFTMatcher
import time


class Lab2App:
    def __init__(self, root):
        self.root = root

        # Устанавливаем начальный заголовок
        self.root.title("Лабораторная работа №2 - SURF/SIFT Обнаружение объектов")

        self.root.geometry("1100x750")
        self.root.resizable(True, True)

        # Инициализация матчера
        self.matcher = SIFTMatcher()

        # Переменные для хранения изображений
        self.template_path = StringVar()
        self.scene_path = StringVar()
        self.template_image = None
        self.scene_image = None
        self.result_image = None
        self.current_frame = None

        # Переменные для камеры
        self.camera_running = False
        self.camera_thread = None
        self.video_capture = None
        self.camera_index = IntVar(value=0)
        self.source_option = StringVar(value="image")  # "image" или "camera"

        # Переменные для отображения
        self.show_keypoints = BooleanVar(value=False)
        self.show_matches = BooleanVar(value=False)

        # Для измерения времени
        self._processing_time = 0.0  # используем защищенную переменную

        # Создание GUI
        self._create_widgets()

        # Привязка событий
        self._bind_events()

        # Обновление состояния
        self._update_state()

        # Обновляем заголовок с начальным временем
        self._update_title()

    @property
    def processing_time(self):
        """Геттер для processor_time"""
        return self._processing_time

    @processing_time.setter
    def processing_time(self, value):
        """Сеттер для processor_time - автоматически обновляет заголовок"""
        self._processing_time = value
        self._update_title()

    def _update_title(self):
        """Обновляет заголовок окна с текущим процессорным временем"""
        self.root.title(
            f"Лабораторная работа №2 - SURF/SIFT Обнаружение объектов "
            f"(Время: {self._processing_time:.6f} с)"
        )

    def _create_widgets(self):
        """Создание всех элементов интерфейса"""

        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        style = ttk.Style()
        style.configure("Custom.TLabelframe.Label", font=("Arial", 10, "bold underline"))

        # Используем стиль при создании
        source_frame = ttk.LabelFrame(
            control_frame,
            text="Выберите источник изображения:",
            padding="5",
            style="Custom.TLabelframe"
        )
        source_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(source_frame, text="Загрузить изображение",
                        variable=self.source_option, value="image",
                        command=self._update_state).pack(anchor=tk.W)

        camera_row = ttk.Frame(source_frame)
        camera_row.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))

        # Переключатель "Включить веб-камеру"
        ttk.Radiobutton(
            camera_row,
            text="Включить веб-камеру",
            variable=self.source_option,
            value="camera",
            command=self._on_camera_toggle
        ).pack(side=tk.LEFT)

        # Отступ
        ttk.Label(camera_row, text="  ").pack(side=tk.LEFT)

        # Выбор камеры - добавляем метку "Камера:"
        ttk.Label(camera_row, text="Камера:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            camera_row,
            from_=0,
            to=10,
            width=5,
            textvariable=self.camera_index
        ).pack(side=tk.LEFT)

        # Выбор изображения сцены
        scene_frame = ttk.LabelFrame(control_frame, text="Выберите изображение сцены:", padding="5")
        scene_frame.pack(fill=tk.X, pady=(0, 10))

        scene_path_frame = ttk.Frame(scene_frame)
        scene_path_frame.pack(fill=tk.X)
        ttk.Entry(scene_path_frame, textvariable=self.scene_path, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(scene_path_frame, text="...", command=self._browse_scene, width=3).pack(side=tk.LEFT, padx=(5, 0))

        # Выбор изображения объекта (шаблона)
        template_frame = ttk.LabelFrame(control_frame, text="Выберите изображение объекта:", padding="5")
        template_frame.pack(fill=tk.X, pady=(0, 10))

        template_path_frame = ttk.Frame(template_frame)
        template_path_frame.pack(fill=tk.X)
        ttk.Entry(template_path_frame, textvariable=self.template_path, width=30).pack(side=tk.LEFT, fill=tk.X,
                                                                                       expand=True)
        ttk.Button(template_path_frame, text="...", command=self._browse_template, width=3).pack(side=tk.LEFT,
                                                                                                 padx=(5, 0))

        # Чекбоксы для отображения
        display_frame = ttk.LabelFrame(control_frame, text="Отображение", padding="5")
        display_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(display_frame, text="Отобразить маркеры",
                        variable=self.show_keypoints,
                        command=self._update_state).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Соединить подобные маркеры",
                        variable=self.show_matches,
                        command=self._update_state).pack(anchor=tk.W)

        # Кнопка загрузки
        self.load_button = ttk.Button(control_frame, text="Загрузить изображение",
                                      command=self._load_and_process)
        self.load_button.pack(fill=tk.X, pady=(0, 10))

        # ===== Правая панель: Отображение изображений =====
        image_frame = ttk.LabelFrame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Канвас для отображения
        self.canvas = tk.Canvas(image_frame, bg='#f0f0f0', width=700, height=550)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Привязка изменения размера
        self.canvas.bind('<Configure>', self._on_resize)

        # Создаем placeholder
        self._show_placeholder()

    def _bind_events(self):
        """Привязка событий"""
        self.root.bind('<Return>', lambda e: self._load_and_process())

    def _show_placeholder(self):
        """Показать заглушку на канвасе"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="Загрузите изображение или включите камеру",
            fill="gray",
            font=("Arial", 14)
        )

    def _browse_scene(self):
        """Выбор изображения сцены"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение сцены",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.scene_path.set(file_path)
            self._update_state()

    def _browse_template(self):
        """Выбор изображения шаблона"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение объекта",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.template_path.set(file_path)
            self._update_state()

    def _on_camera_toggle(self):
        """Переключение камеры при выборе опции"""
        if self.source_option.get() == "camera":
            self._start_camera()
        else:
            self._stop_camera()

    def _load_and_process(self):
        """Загрузка и обработка изображений"""
        if self.source_option.get() == "camera":
            return

        if not self.scene_path.get():
            messagebox.showwarning("Предупреждение", "Выберите изображение сцены!")
            return

        if not self.template_path.get():
            messagebox.showwarning("Предупреждение", "Выберите изображение объекта!")
            return

        # Загружаем шаблон
        self._update_status("Загрузка шаблона...", "orange")
        if not self.matcher.load_template(self.template_path.get()):
            self._update_status("Ошибка загрузки шаблона!", "red")
            messagebox.showerror("Ошибка", "Не удалось загрузить шаблон")
            return

        # Загружаем сцену
        self._update_status("Загрузка сцены...", "orange")
        self.scene_image = cv2.imread(self.scene_path.get())
        if self.scene_image is None:
            self._update_status("Ошибка загрузки сцены!", "red")
            messagebox.showerror("Ошибка", "Не удалось загрузить сцену")
            return

        # Выполняем поиск
        self._update_status("Поиск объекта...", "orange")
        self.root.config(cursor="watch")
        self.root.update()

        start_time = time.time()
        result, output_img = self.matcher.find_object(
            self.scene_image,
            draw_keypoints=self.show_keypoints.get(),
            draw_matches=self.show_matches.get()
        )
        self.processing_time = time.time() - start_time  # автоматически обновит заголовок

        self.root.config(cursor="")

        # Сохраняем результат
        self.result_image = output_img
        self._show_image(output_img)

        # Обновляем статус
        if result.success:
            self._update_status(f"✅ Объект найден! Совпадений: {result.match_count}", "green")
        else:
            self._update_status(f"❌ {result.message}", "red")

    def _start_camera(self):
        """Запуск веб-камеры"""
        if self.camera_running:
            return

        if not self.template_path.get():
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение объекта!")
            return

        # Загружаем шаблон
        if not self.matcher.load_template(self.template_path.get()):
            messagebox.showerror("Ошибка", "Не удалось загрузить шаблон")
            return

        try:
            self.video_capture = cv2.VideoCapture(self.camera_index.get())
            if not self.video_capture.isOpened():
                raise Exception("Не удалось открыть камеру")

            self.camera_running = True
            self._update_status("Камера запущена. Поиск объекта...", "green")

            # Запускаем поток обработки
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            self._update_status(f"Ошибка камеры: {e}", "red")
            messagebox.showerror("Ошибка", f"Не удалось запустить камеру:\n{e}")

    def _camera_loop(self):
        """Цикл обработки кадров с камеры"""
        frame_count = 0
        while self.camera_running:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                frame_count += 1
                # Обрабатываем каждый 2-й кадр для экономии ресурсов
                if frame_count % 2 == 0:
                    continue

                # Поиск объекта на кадре
                start_time = time.time()
                result, output_img = self.matcher.find_object(
                    frame,
                    draw_keypoints=self.show_keypoints.get(),
                    draw_matches=self.show_matches.get()
                )
                # Обновляем время (автоматически обновит заголовок)
                self.processing_time = time.time() - start_time

                # Обновляем изображение в главном потоке
                self.root.after(0, lambda img=output_img: self._show_image(img))

                # Обновляем статус
                if result.success:
                    self.root.after(0, lambda: self._update_status(
                        f"✅ Объект найден! Совпадений: {result.match_count}", "green"
                    ))
                else:
                    self.root.after(0, lambda: self._update_status(
                        f"⏳ Поиск...", "orange"
                    ))

            except Exception as e:
                print(f"Ошибка в цикле камеры: {e}")
                break

    def _stop_camera(self):
        """Остановка камеры"""
        self.camera_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self._update_status("Камера остановлена", "gray")
        self._show_placeholder()

    def _show_image(self, image):
        """Отображение изображения на канвасе"""
        if image is None:
            self._show_placeholder()
            return

        # Получаем размеры канваса
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 700
            canvas_height = 550

        # Конвертируем BGR в RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Масштабируем с сохранением пропорций
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
        else:
            image_resized = image_rgb
            new_w, new_h = w, h

        # Конвертируем в PhotoImage
        image_pil = Image.fromarray(image_resized)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Отображаем
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=image_tk
        )
        self.canvas.image = image_tk  # Сохраняем ссылку

    def _on_resize(self, event):
        """Обработка изменения размера канваса"""
        if self.result_image is not None:
            self._show_image(self.result_image)
        elif self.scene_image is not None:
            self._show_image(self.scene_image)

    def _update_status(self, message, color="black"):
        """Обновление статуса"""
        self.root.update_idletasks()

    def _update_state(self):
        """Обновление состояния элементов управления"""
        is_camera = self.source_option.get() == "camera"

        # Включаем/выключаем кнопки
        if is_camera:
            self.load_button.config(state=tk.DISABLED)
        else:
            self.load_button.config(state=tk.NORMAL)

    def on_closing(self):
        """Обработка закрытия окна"""
        self._stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = Lab2App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()