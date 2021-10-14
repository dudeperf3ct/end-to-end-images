from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import image_predictor_router

app = FastAPI()
app.include_router(image_predictor_router.router, prefix="/Image")

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:*",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'dummy check! classifier  is all ready to go!'
