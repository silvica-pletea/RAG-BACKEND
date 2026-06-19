from fastapi import UploadFile
import shutil
from app.config.settings import SAVE_FILES_PATH

class FileService:
    
    async def save_file(
        self,
        file: UploadFile,
    ) -> str:
        SAVE_FILES_PATH.mkdir(parents=True, exist_ok=True)
        file_path = SAVE_FILES_PATH / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return str(file_path)

    def get_files(self) -> list[str]:
        if not SAVE_FILES_PATH.exists():
            return []
        return [f.name for f in SAVE_FILES_PATH.iterdir() if f.is_file()]
    
    def delete_file(self, file_name: str) -> None:
        path = SAVE_FILES_PATH / file_name
        if path.exists():
            path.unlink()
