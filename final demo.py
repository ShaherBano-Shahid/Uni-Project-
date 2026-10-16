
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# BOOK COVER RECOGNITION - PROJECT SETTINGS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "prepared_data"
    / "book_cover_mobilenet_finetuned.keras"
)

IMG_SIZE = (224, 224)
NUM_CLASSES = 30

# Dataset details
TRAIN_IMAGES = 46170
VALIDATION_IMAGES = 5130
TEST_IMAGES = 5700
TOTAL_IMAGES = 57000

# Evaluation results
CNN_TEST_ACCURACY = 6.35
MOBILENET_TRAIN_ACCURACY = 28.41
MOBILENET_VALIDATION_ACCURACY = 26.10
FINETUNED_VALIDATION_ACCURACY = 28.09
FINETUNED_TEST_ACCURACY = 26.79
MACRO_F1 = 25.27
WEIGHTED_F1 = 25.27

# =====================================================
# CLASS ID TO CATEGORY NAME
# Verify this order against the official dataset labels.
# =====================================================

CLASS_NAMES = {
    0: "Arts & Photography",
    1: "Biographies & Memoirs",
    2: "Business & Money",
    3: "Calendars",
    4: "Children's Books",
    5: "Comics & Graphic Novels",
    6: "Computers & Technology",
    7: "Cookbooks, Food & Wine",
    8: "Crafts, Hobbies & Home",
    9: "Christian Books & Bibles",
    10: "Engineering & Transportation",
    11: "Health, Fitness & Dieting",
    12: "History",
    13: "Humor & Entertainment",
    14: "Law",
    15: "Literature & Fiction",
    16: "Medical Books",
    17: "Mystery, Thriller & Suspense",
    18: "Parenting & Relationships",
    19: "Politics & Social Sciences",
    20: "Reference",
    21: "Religion & Spirituality",
    22: "Romance",
    23: "Science & Math",
    24: "Science Fiction & Fantasy",
    25: "Self-Help",
    26: "Sports & Outdoors",
    27: "Teen & Young Adult",
    28: "Test Preparation",
    29: "Travel"
}


# =====================================================
# LOAD MODEL
# =====================================================

model = None

if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        if model.output_shape[-1] != NUM_CLASSES:
            messagebox_error = (
                f"Expected {NUM_CLASSES} output classes, "
                f"but model has {model.output_shape[-1]}."
            )
            raise ValueError(messagebox_error)

    except Exception as error:
        print("Model loading error:", error)
else:
    print("Model not found:", MODEL_PATH)


# =====================================================
# CREATE WINDOW
# =====================================================

root = tk.Tk()
root.title("Book Cover Recognition")
root.geometry("760x620")
root.minsize(700, 560)
root.configure(bg="#f1f5f9")

style = ttk.Style()
style.theme_use("clam")
style.configure(
    "TNotebook.Tab",
    padding=(12, 7),
    font=("Arial", 9, "bold")
)

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=8)

overview_tab = tk.Frame(notebook, bg="#f1f5f9")
results_tab = tk.Frame(notebook, bg="#f1f5f9")
prediction_tab = tk.Frame(notebook, bg="#f1f5f9")

notebook.add(overview_tab, text="Project Overview")
notebook.add(results_tab, text="Milestones & Results")
notebook.add(prediction_tab, text="Predict Book Cover")

# =====================================================
# PROJECT OVERVIEW
# =====================================================

tk.Label(
    overview_tab,
    text="BOOK COVER RECOGNITION",
    font=("Arial", 20, "bold"),
    bg="#f1f5f9",
    fg="#17365d"
).pack(pady=(20, 5))

tk.Label(
    overview_tab,
    text="Machine Learning | Deep Learning | Computer Vision",
    font=("Arial", 10),
    bg="#f1f5f9",
    fg="#475569"
).pack(pady=(0, 15))

overview_card = tk.Frame(
    overview_tab,
    bg="white",
    padx=18,
    pady=15,
    highlightbackground="#dbe3ed",
    highlightthickness=1
)
overview_card.pack(fill="x", padx=15, pady=6)

tk.Label(
    overview_card,
    text="Project Purpose",
    font=("Arial", 12, "bold"),
    bg="white",
    fg="#17365d"
).pack(anchor="w")

tk.Label(
    overview_card,
    text=(
        "This project predicts the category of a book cover from an image. "
        "It uses the BookCover30 dataset and a MobileNetV2 deep-learning model."
    ),
    font=("Arial", 10),
    bg="white",
    wraplength=650,
    justify="left"
).pack(anchor="w", pady=(6, 12))

tk.Label(
    overview_card,
    text="Dataset Information",
    font=("Arial", 12, "bold"),
    bg="white",
    fg="#17365d"
).pack(anchor="w")

tk.Label(
    overview_card,
    text=(
        f"Training: {TRAIN_IMAGES:,} images\n"
        f"Validation: {VALIDATION_IMAGES:,} images\n"
        f"Testing: {TEST_IMAGES:,} images\n"
        f"Total: {TOTAL_IMAGES:,} images | Classes: {NUM_CLASSES}"
    ),
    font=("Arial", 10),
    bg="white",
    justify="left"
).pack(anchor="w", pady=(6, 12))

tk.Label(
    overview_card,
    text="Technologies Used",
    font=("Arial", 12, "bold"),
    bg="white",
    fg="#17365d"
).pack(anchor="w")

tk.Label(
    overview_card,
    text=(
        "• Computer Vision: image resizing and preprocessing\n"
        "• Deep Learning: CNN and MobileNetV2\n"
        "• Transfer Learning and Fine-tuning\n"
        "• Python Tkinter graphical interface"
    ),
    font=("Arial", 10),
    bg="white",
    justify="left"
).pack(anchor="w", pady=(6, 0))

# =====================================================
# MILESTONES AND RESULTS
# =====================================================

tk.Label(
    results_tab,
    text="PROJECT MILESTONES",
    font=("Arial", 17, "bold"),
    bg="#f1f5f9",
    fg="#17365d"
).pack(anchor="w", padx=15, pady=(15, 5))

milestone_text = (
    "1. Dataset Selection\n"
    "Selected BookCover30 with 30 classes.\n\n"
    "2. Data Preparation\n"
    "Prepared 46,170 training, 5,130 validation and "
    "5,700 test images.\n\n"
    "3. Feature Extraction\n"
    "MobileNetV2 feature map: (7, 7, 1280). "
    "Global Average Pooling produces 1,280 features.\n\n"
    "4. CNN Baseline\n"
    f"Test accuracy: {CNN_TEST_ACCURACY:.2f}%.\n\n"
    "5. MobileNetV2 Training\n"
    f"Training accuracy: {MOBILENET_TRAIN_ACCURACY:.2f}%.\n"
    f"Validation accuracy: {MOBILENET_VALIDATION_ACCURACY:.2f}%.\n\n"
    "6. Fine-tuning\n"
    f"Validation accuracy: {FINETUNED_VALIDATION_ACCURACY:.2f}%.\n\n"
    "7. Final Test Evaluation\n"
    f"Test accuracy: {FINETUNED_TEST_ACCURACY:.2f}%.\n"
    f"Macro F1-score: {MACRO_F1:.2f}%.\n"
    f"Weighted F1-score: {WEIGHTED_F1:.2f}%.\n\n"
    "Next step: verify class mapping and analyse incorrect predictions."
)

milestone_box = tk.Text(
    results_tab,
    font=("Arial", 10),
    bg="white",
    fg="#263445",
    wrap="word",
    padx=12,
    pady=12,
    height=25
)
milestone_box.pack(fill="both", expand=True, padx=15, pady=8)
milestone_box.insert("1.0", milestone_text)
milestone_box.config(state="disabled")

# =====================================================
# PREDICTION TAB
# =====================================================

left_frame = tk.Frame(prediction_tab, bg="#f1f5f9")
left_frame.pack(side="left", fill="both", expand=True, padx=8, pady=10)

right_frame = tk.Frame(prediction_tab, bg="#f1f5f9")
right_frame.pack(side="right", fill="both", expand=True, padx=8, pady=10)

image_card = tk.Frame(
    left_frame,
    bg="white",
    padx=10,
    pady=10,
    highlightbackground="#dbe3ed",
    highlightthickness=1
)
image_card.pack(fill="both", expand=True)

tk.Label(
    image_card,
    text="Book Cover Preview",
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#17365d"
).pack(pady=(0, 8))

image_label = tk.Label(
    image_card,
    text="Upload a book cover",
    bg="white",
    fg="#64748b"
)
image_label.pack(fill="both", expand=True)

filename_label = tk.Label(
    image_card,
    text="No image selected",
    font=("Arial", 8),
    bg="white",
    fg="#64748b",
    wraplength=280
)
filename_label.pack(pady=5)


result_card = tk.Frame(
    right_frame,
    bg="white",
    padx=12,
    pady=12,
    highlightbackground="#dbe3ed",
    highlightthickness=1
)
result_card.pack(fill="both", expand=True)

tk.Label(
    result_card,
    text="Prediction Result",
    font=("Arial", 12, "bold"),
    bg="white",
    fg="#17365d"
).pack(anchor="w")

result_label = tk.Label(
    result_card,
    text="Upload an image to begin",
    font=("Arial", 13, "bold"),
    bg="white",
    fg="#17365d",
    wraplength=300,
    justify="center"
)
result_label.pack(pady=16)

score_label = tk.Label(
    result_card,
    text="",
    font=("Arial", 10),
    bg="white",
    fg="#15803d"
)
score_label.pack(pady=5)

top_label = tk.Label(
    result_card,
    text="",
    font=("Consolas", 9),
    bg="white",
    fg="#334155",
    justify="left"
)
top_label.pack(pady=10, anchor="w")
selected_path = None
preview_photo = None

# =====================================================
# UPLOAD IMAGE
# =====================================================

def upload_image():
    global selected_path, preview_photo

    path = filedialog.askopenfilename(
        title="Select a Book Cover",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return

    try:
        image = Image.open(path).convert("RGB")
        selected_path = path

        image.thumbnail((270, 330))
        preview_photo = ImageTk.PhotoImage(image)

        image_label.config(image=preview_photo, text="")
        filename_label.config(text=Path(path).name)

        result_label.config(
            text="Ready to predict",
            fg="#17365d"
        )
        score_label.config(text="")
        top_label.config(text="")

    except Exception as error:
        messagebox.showerror("Image Error", str(error))

# =====================================================
# PREDICT CATEGORY
# =====================================================

def predict_category():
    if model is None:
        messagebox.showerror(
            "Model Error",
            "The trained model could not be loaded.\n\n"
            f"Expected file:\n{MODEL_PATH}"
        )
        return

    if selected_path is None:
        messagebox.showwarning(
            "No Image",
            "Please upload a book cover first."
        )
        return

    try:
        image = Image.open(selected_path).convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(image, dtype=np.float32)
        array = preprocess_input(array)
        array = np.expand_dims(array, axis=0)

        predictions = model.predict(array, verbose=0)[0]

        predicted_class = int(np.argmax(predictions))
        genre_name = CLASS_NAMES.get(
            predicted_class,
            f"Unknown Category (Class {predicted_class})"
        )

        confidence = float(predictions[predicted_class]) * 100

        result_label.config(
            text=(
                f"Predicted Category:\n{genre_name}\n\n"
                f"Class ID: {predicted_class}"
            ),
            fg="#15803d"
        )

        score_label.config(
            text=f"Model Score: {confidence:.2f}%"
        )

        top_3 = np.argsort(predictions)[-3:][::-1]

        lines = ["Top 3 Predictions:"]
        for class_id in top_3:
            class_id = int(class_id)
            name = CLASS_NAMES.get(class_id, "Unknown")
            score = float(predictions[class_id]) * 100
            lines.append(f"{name}: {score:.1f}%")

        top_label.config(text="\n".join(lines))

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            str(error)
        )

# =====================================================
# CLEAR IMAGE AND RESULT
# =====================================================

def clear_all():
    global selected_path, preview_photo

    selected_path = None
    preview_photo = None

    image_label.config(
        image="",
        text="Upload a book cover"
    )
    filename_label.config(text="No image selected")
    result_label.config(text="Upload an image to begin")
    score_label.config(text="")
    top_label.config(text="")

# =====================================================
# BUTTONS
# =====================================================

button_frame = tk.Frame(right_frame, bg="white")
button_frame.pack(fill="x", pady=(12, 0))

tk.Button(
    button_frame,
    text="Upload",
    command=upload_image,
    bg="#2563eb",
    fg="white",
    font=("Arial", 9, "bold"),
    relief="flat",
    padx=8,
    pady=8
).pack(side="left", padx=3)

tk.Button(
    button_frame,
    text="Predict",
    command=predict_category,
    bg="#15803d",
    fg="white",
    font=("Arial", 9, "bold"),
    relief="flat",
    padx=8,
    pady=8
).pack(side="left", padx=3)

tk.Button(
    button_frame,
    text="Clear",
    command=clear_all,
    bg="#475569",
    fg="white",
    font=("Arial", 9, "bold"),
    relief="flat",
    padx=8,
    pady=8
).pack(side="left", padx=3)

# =====================================================
# STATUS BAR
# =====================================================

if model is not None:
    status_text = f"Model loaded: {MODEL_PATH.name}"
else:
    status_text = "Model not loaded - check the model file path"

tk.Label(
    root,
    text=status_text,
    font=("Arial", 8),
    bg="#e2e8f0",
    fg="#334155",
    anchor="w",
    padx=10,
    pady=5
).pack(fill="x", side="bottom")
root.mainloop()
