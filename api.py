from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "PaperScraper API is running"}


@app.post("/upload-test")
async def upload_test(file: UploadFile):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
    }
