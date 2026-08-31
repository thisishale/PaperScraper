import os
import tempfile
import time

from fastapi import FastAPI, Request, UploadFile

from main import extract

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # call_next() is what actually runs the matching route (read_root,
    # upload_test, extract_endpoint, whichever one matches this request) --
    # everything before it runs before the route, everything after runs
    # after, for every single route, without editing any of them.
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    print(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")

    return response


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
async def extract_endpoint(file: UploadFile):
    # Now async -> back to the async .read() (like /upload-test), and
    # extract() is awaited directly since it's a real coroutine now, not
    # wrapped in asyncio.run() the way main.py's sync batch script needs.
    contents = await file.read()

    # extract() takes a file path, not raw bytes -- unstructured's
    # partition_pdf needs to open the file itself. Write to a temp file,
    # then delete it once extract() is done (even if it raises).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = await extract(tmp_path)
    finally:
        os.remove(tmp_path)

    return result
