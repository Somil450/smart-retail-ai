from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
from app.services.cv_service import recognize_face, classify_product

router = APIRouter(prefix="/vision", tags=["Computer Vision"])

@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def api_recognize_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    image_bytes = await file.read()
    result = recognize_face(image_bytes)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return FaceRecognitionResponse(**result)

@router.post("/classify-product", response_model=ProductClassificationResponse)
async def api_classify_product(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    image_bytes = await file.read()
    result = classify_product(image_bytes)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return ProductClassificationResponse(**result)
