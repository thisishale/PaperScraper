import os
import tempfile

from fastapi import FastAPI, UploadFile

from main import extract

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


@app.post("/extract")
def extract_endpoint(file: UploadFile):
    # Sync route -> use the sync .file attribute, not the async .read() we
    # used in /upload-test. UploadFile exposes both; which one you use
    # depends on whether the route itself is async def or plain def.
    contents = file.file.read()

    # extract() takes a file path, not raw bytes -- unstructured's
    # partition_pdf needs to open the file itself. Write to a temp file,
    # then delete it once extract() is done (even if it raises).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = extract(tmp_path)
    finally:
        os.remove(tmp_path)

    return result
