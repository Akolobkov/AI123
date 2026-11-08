import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import easyocr
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger('easyocr').setLevel(logging.ERROR)


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР3 — Распознавание текста (EasyOCR)")
        self.root.geometry("1100x700")

        # --- Инициализация EasyOCR (один раз) ---
        print("Загрузка моделей EasyOCR...")
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        print("Готово.")

        self.current_image = None       # оригинал (BGR)
        self.result_image = None        # с обведённым текстом (BGR)
        self.tk_image = None            # для отображения в Label

        self._build_ui()

    def _build_ui(self):
        top_frame = tk.Frame(self.root, pady=8)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load = tk.Button(top_frame, text="📂 Загрузить изображение",
                             command=self.load_image, font=("Arial", 11))
        btn_load.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(top_frame, text="Выберите изображение",
                                     font=("Arial", 11), fg="gray")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Основная область: картинка слева, текст справа
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая часть — изображение
        self.image_label = tk.Label(main_frame, bg="#222", text="Здесь будет изображение",
                                    fg="white", font=("Arial", 12))
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Правая часть — распознанный текст
        right_frame = tk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Распознанный текст:",
                 font=("Arial", 11, "bold")).pack(anchor="w")

        text_frame = tk.Frame(right_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.text_output = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.text_output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=scrollbar.set)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("Все файлы", "*.*")]
        )
        if not path:
            return

        # Читаем через OpenCV (для не-латинских путей используем imdecode)
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Не удалось прочитать изображение")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.current_image = image
        self.status_label.config(text="⏳ Распознавание...", fg="orange")
        self.root.update()

        self.run_ocr(path)

    def run_ocr(self, path):
        try:
            # EasyOCR умеет читать файл сам (в т.ч. кириллические пути)
            results = self.reader.readtext(path, paragraph=False)
        except Exception as e:
            messagebox.showerror("Ошибка OCR", str(e))
            self.status_label.config(text="Ошибка", fg="red")
            return

        # --- Отрисовка ---
        result_image = self.current_image.copy()
        lines = []

        for (bbox, text, prob) in results:
            if prob < 0.3:
                continue

            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            # Рамка
            pts = np.array([tl, tr, br, bl], dtype=np.int32)
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)

            # Подпись с уверенностью
            label = f"{prob:.2f}"
            cv2.putText(result_image, label, (tl[0], max(tl[1] - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            lines.append(f"[{prob:.2f}] {text}")

        self.result_image = result_image
        # --- Текст в панель ---
        self.text_output.delete("1.0", tk.END)
        if lines:
            self.text_output.insert(tk.END, "\n".join(lines))
            self.status_label.config(
                text=f"✅ Найдено строк: {len(lines)}", fg="green")
        else:
            self.text_output.insert(tk.END, "(Текст не найден)")
            self.status_label.config(text="⚠️ Текст не найден", fg="orange")

        # --- Показ картинки ---
        self.show_image(result_image)

    def show_image(self, bgr_image):
        # Масштабируем под размер Label, сохраняя пропорции
        h, w = bgr_image.shape[:2]
        label_w = max(self.image_label.winfo_width(), 400)
        label_h = max(self.image_label.winfo_height(), 400)
        scale = min(label_w / w, label_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(bgr_image, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.tk_image, text="")

    def save_result(self):
        if self.result_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if not path:
            return
        # Сохраняем через imencode, чтобы работало с кириллицей в пути
        ext = "." + path.split(".")[-1]
        success, buf = cv2.imencode(ext, self.result_image)
        if success:
            buf.tofile(path)
            messagebox.showinfo("Сохранено", f"Файл сохранён:\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()