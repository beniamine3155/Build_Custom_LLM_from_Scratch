from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import uvicorn
import os
import json

# Initialize FastAPI app
app = FastAPI(title="Custom LLM API")

class TextInput(BaseModel):
    text: str

# Global variable
model_path = None
classifier = None
label_mapping = None


# Load the fine-tuned model 
def find_and_load_model():
    global model_path, classifier, label_mapping

    model_path = [
        "models/custom_support_model_WORKING"
    ]

    for path in model_path:
        if os.path.exists(path):
            try:
                classifier = pipeline("text-classification", model=path)
                model_path = path


                # load label mapping
                label_file = os.path.join(path, "label_mapping.json")
                if os.path.exists(label_file):
                    with open(label_file, "r") as f:
                        mapping_data = json.load(f)
                        if label_mapping in mapping_data:
                            label_mapping = mapping_data['label_mapping']
                        else:
                            label_mapping = mapping_data
                print(f"Model loaded successfully from {path}")
                return True
            
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                continue
    print("No model found.")
    return False


@app.on_event("startup")
async def startup():
    find_and_load_model()


@app.get("/", response_class=HTMLResponse)
async def web_interface():
    model_status = "Model loaded successfully." if classifier else "No model loaded."
    model_info = f"Path: {model_path}" if model_path else "Train a model first."
    categories = list(label_mapping.keys()) if label_mapping else ["adult_explicit", "hateful_harmful", "violent_graphic","spam_promotional","safe_neutral"]
    html_content = f"""
    <html>
        <head>
            <title>Custom LLM API</title>
        </head>
        <body>
            <h1>Welcome to the Custom LLM API</h1>
            <form action="/predict" method="post">
                <textarea name="text" rows="10" cols="50" placeholder="Enter text here..."></textarea><br>
                <input type="submit" value="Classify">
            </form>
        </body>
    </html>
    """
    return html_content


# API endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "model_path": model_path 
    }


@app.post("/classify")
async def classify(input: TextInput):
    if not classifier:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train a model first.")
    try:
        result = classifier(input.text)
        if isinstance(result, list):
            result = result[0]

        return {
            "text": input.text,
            "category": result.get("label", "unknown"),
            "confidence": result.get("score", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during classification: {e}")
    


        