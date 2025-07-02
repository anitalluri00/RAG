from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import os
from pathlib import Path

# Create static directory if it doesn't exist
Path("static").mkdir(exist_ok=True)

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize vectorstore and retriever
vectorstore = None
retriever = None
qa_chain = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    global vectorstore, retriever, qa_chain
    
    try:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        docs = process_file(file_path)
        vectorstore, retriever = get_retriever(docs)
        qa_chain = get_qa_chain(retriever)
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": f"File {file.filename} uploaded and processed successfully!"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error processing file: {str(e)}"
            }
        )

@app.post("/query")
async def handle_query(request: Request, query: str = Form(...)):
    global qa_chain
    
    if not qa_chain:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Please upload a file first"
            }
        )
    
    try:
        result = qa_chain.invoke({"query": query})
        return templates.TemplateResponse(
            "response.html",
            {
                "request": request,
                "query": query,
                "result": result["result"]
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error processing query: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
