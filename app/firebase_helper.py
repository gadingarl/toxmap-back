import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime


if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

async def save_scan_result(user_id, result_label, dropbox_color, image_url=""):
    """
    Simpan hasil scan ke koleksi scan_history di Firestore.
    """
    scan_id = str(uuid.uuid4())
    doc_ref = db.collection("scan_history").document(scan_id)
    doc_ref.set({
        "user_id": user_id,
        "result": result_label,
        "dropbox_color": dropbox_color,
        "timestamp": datetime.utcnow(),
        "image_url": image_url
    })
    return scan_id
