import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid

# Inisialisasi Firebase hanya sekali
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'toxmap-b74f4.firebasestorage.app'  # SESUAI UI Firebase kamu
    })

# Firestore client
db = firestore.client()

def save_scan_result(user_id, result_label, dropbox_color, image_url=""):
    """
    Simpan hasil scan ke koleksi scan_history di Firestore.
    """
    scan_id = str(uuid.uuid4())
    doc_ref = db.collection("scan_history").document(scan_id)
    doc_ref.set({
        "user_id": user_id,
        "result": result_label,
        "dropbox_color": dropbox_color,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "image_url": image_url
    })
    return scan_id

def upload_image_to_storage(file_bytes, filename):
    """
    Upload gambar ke Firebase Storage (scan_images/) dan kembalikan public URL-nya.
    """
    bucket = storage.bucket()
    blob = bucket.blob(f"scan_images/{filename}")
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    blob.make_public()  # supaya bisa diakses oleh frontend (Flutter)
    return blob.public_url
