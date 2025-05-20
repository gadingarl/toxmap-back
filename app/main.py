from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from app.utils import predict_image
from app.firebase_helper import save_scan_result  # Panggil helper baru
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS supaya bisa diakses Flutter/web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File yang diizinkan
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

@app.get("/")
async def root():
    return {"message": "TOXMAP Backend Ready!"}

@app.post("/predict/")
async def predict(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Validasi ekstensi
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="File type not allowed. Only jpg, jpeg, png accepted."
            )

        file_data = await file.read()

        # Prediksi
        result_label, dropbox_color = predict_image(file_data)

        # Dummy image_url (nanti kalau upload storage baru diganti)
        image_url = ""

        # Simpan hasil ke Firestore
        await save_scan_result(
            user_id=user_id,
            result_label=result_label,
            dropbox_color=dropbox_color,
            image_url=image_url
        )

        return {
            "result": result_label,
            "dropbox_color": dropbox_color,
            "image_url": image_url
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )
