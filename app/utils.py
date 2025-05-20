import joblib
import numpy as np
from PIL import Image
import io

# Load model
model = joblib.load('svm_model.pkl')

# Mapping label angka ke nama kategori
label_mapping = {
    0: "Non_Toxic",
    1: "Baterai",
    2: "Kabel",
    3: "LampuLED",
    4: "Aerosol",
    5: "PembersihLantai"
}

def predict_image(file: bytes):
    # Baca & proses gambar
    image = Image.open(io.BytesIO(file)).convert('L')  # convert ke grayscale
    image = image.resize((90, 90))  # Sesuaikan ke 90x90 sesuai model
    img_array = np.array(image)
    img_flat = img_array.flatten().reshape(1, -1)

    # Prediksi
    pred = model.predict(img_flat)
    pred_label = label_mapping.get(pred[0], "Unknown")

    # Mapping warna dropbox
    dropbox_color_mapping = {
        "Baterai": "Merah",
        "Kabel": "Merah",
        "LampuLED": "Merah",
        "Aerosol": "Kuning",
        "PembersihLantai": "Kuning",
        "Non_Toxic": "Tidak Ada"
    }

    dropbox_color = dropbox_color_mapping.get(pred_label, "Tidak Ada")

    return pred_label, dropbox_color

