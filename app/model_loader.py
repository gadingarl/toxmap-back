from PIL import Image
from io import BytesIO
import numpy as np
import joblib
import math

model = joblib.load("svm_model_final.pkl")
n_features = model.support_vectors_.shape[1]

# Hitung ukuran gambar dari total fitur
# Karena RGB, maka total pixel = fitur / 3
total_pixel = n_features // 3

# Cari kombinasi tinggi dan lebar paling mendekati bujur sangkar
def get_closest_hw(pixel_count):
    sqrt_val = int(math.sqrt(pixel_count))
    for h in range(sqrt_val, 0, -1):
        if pixel_count % h == 0:
            w = pixel_count // h
            return h, w
    return 1, pixel_count

target_h, target_w = get_closest_hw(total_pixel)
print(f"📐 Model expect shape: ({target_h}, {target_w}, 3) → Total fitur: {n_features}")

label_mapping = {
    0: "Non_Toxic",
    1: "Baterai",
    2: "Kabel",
    3: "LampuLED",
    4: "Aerosol",
    5: "PembersihLantai"
}

dropbox_color_mapping = {
    "Baterai": "Merah",
    "Kabel": "Merah",
    "LampuLED": "Merah",
    "Aerosol": "Kuning",
    "PembersihLantai": "Kuning",
    "Non_Toxic": "Tidak Ada"
}

def predict_image(file_bytes):
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        image = image.resize((target_w, target_h))  # width x height

        img_array = np.asarray(image, dtype=np.uint8)
        if img_array.shape != (target_h, target_w, 3):
            raise ValueError(f"❌ Shape salah: {img_array.shape}, harus ({target_h}, {target_w}, 3)")

        flat = img_array.flatten().reshape(1, -1)
        if flat.shape[1] != n_features:
            raise ValueError(f"❌ Jumlah fitur salah: {flat.shape[1]} ≠ {n_features}")

        prediction = model.predict(flat)[0]
        pred_label = label_mapping.get(prediction, "Unknown")
        dropbox_color = dropbox_color_mapping.get(pred_label, "Tidak Ada")

        return pred_label, dropbox_color

    except Exception as e:
        raise ValueError(f"❌ Gagal memproses gambar: {str(e)}")
