from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import files, chat

app = FastAPI()

list = ["http://localhost:8000",
        "http://localhost:4200",
        "https://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)
app.include_router(chat.router)