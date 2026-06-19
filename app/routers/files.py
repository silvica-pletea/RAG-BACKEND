from fastapi import APIRouter, HTTPException, UploadFile
from pathlib import Path
from app.services.file_service import FileService
from app.services.embedding_service import EmbeddingService
from app.config.settings import ALLOWED_TYPE, MAX_FILE_SIZE_BYTES


router = APIRouter(prefix="/files", tags=["files"])
file_service = FileService()
embedding_service = EmbeddingService()

@router.post("/upload")
async def upload_file(file: UploadFile):

    # Check if user selected a file
    if not file.filename or file.filename.strip() == "":
        raise HTTPException(
            status_code = 400,
            detail = {
                "message": "File validation failed",
                "errors": 'No file selected'
            }
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext != ALLOWED_TYPE:
        raise HTTPException(
            status_code = 400, 
            detail = {
                "message": "File validation failed",
                "errors": f"File extension '{file_ext}' not allowed. Use a .pdf file."
            }
        )

    # Get the file size (in bytes)
    file.file.seek(0, 2)
    file_size = file.file.tell()

    # Move the cursor back to the beginning
    await file.seek(0)

    # check the file size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail = {
                "message": "File size exceeds the limit",
                "errors": f"File too large ({file_size:,} bytes). Maximum: {MAX_FILE_SIZE_BYTES:,} bytes"
            }
        )

    # Save locally
    try:
        file_path = await file_service.save_file(file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail= {
                "message": "Failed to save file",
                "errors": f"Failed to save file: {str(e)}"
            }
    )

    # This should be a worker or chron job to follow the RAG flow
    try: 
        embedding_service.process_file(file_path, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail= {
                "message": "Failed to embedding file",
                "errors": f"Failed to embedding file: {str(e)}"
            }
        )
    return "File uploaded successfully!"

@router.get("/")
def get_files():
    try:
        files = file_service.get_files()
        return { "files": files, "total": len(files) }
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = {
                "message": "Failed to list files",
                "errors": str(e)
            }
        )

@router.delete("/delete/{file_name}")
async def delete(file_name):
    try: 
        embedding_service.delete_file(file_name)
        file_service.delete_file(file_name)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = {
                "message": "Failed to delete a file",
                "errors": str(e)
            }
        )