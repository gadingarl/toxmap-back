from PIL import Image
from io import BytesIO
import numpy as np
import joblib

# Load model SVM
model = joblib.load('svm_model.pkl')  # pastikan path-nya benar

async def predict_image(file):
    try:
        # Baca file gambar
        file_data = await file.read()

        # Buka sebagai gambar dan convert ke grayscale
        image = Image.open(BytesIO(file_data)).convert('L')

        # Resize ke 128x128 (harus sama kayak training!)
        image = image.resize((128, 128))

        # Konversi ke numpy array dan flatten
        data = np.asarray(image, dtype=float).flatten()

        # Normalisasi
        data = data / 255.0

        # Prediksi
        prediction = model.predict([data])[0]

        # Mapping label (disesuaikan dengan label kamu ya)
        if prediction == 1:
            result = "B3"
            dropbox_color = "Merah"
        else:
            result = "Non-B3"
            dropbox_color = "Hijau"

        return result, dropbox_color

    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")
