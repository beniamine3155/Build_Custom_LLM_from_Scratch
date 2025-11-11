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
    <!DOCTYPE html>
    <html>
    <head>
        <title>Content Safety Classifier</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                min-height: 100vh;
                padding: 40px 20px;
            }}
            
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                padding: 40px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 2px solid #f8f9fa;
                padding-bottom: 20px;
            }}
            
            .header h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.2em;
                font-weight: 300;
            }}
            
            .header p {{
                color: #7f8c8d;
                font-size: 1.1em;
            }}
            
            .status-section {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 30px;
                border-left: 4px solid #3498db;
            }}
            
            .status-item {{
                margin-bottom: 10px;
            }}
            
            .status-label {{
                font-weight: 600;
                color: #2c3e50;
                display: inline-block;
                width: 120px;
            }}
            
            .status-value {{
                color: #34495e;
            }}
            
            .input-section {{
                margin-bottom: 30px;
            }}
            
            .input-label {{
                display: block;
                margin-bottom: 12px;
                font-weight: 600;
                color: #2c3e50;
                font-size: 1.1em;
            }}
            
            .text-input {{
                width: 100%;
                padding: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                font-size: 1em;
                resize: vertical;
                min-height: 140px;
                font-family: 'Arial', sans-serif;
                transition: border-color 0.3s ease;
                background: #fff;
            }}
            
            .text-input:focus {{
                outline: none;
                border-color: #3498db;
                background: #fafafa;
            }}
            
            .button-group {{
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
            }}
            
            .btn {{
                padding: 14px 28px;
                border: none;
                border-radius: 6px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                flex: 1;
            }}
            
            .btn-primary {{
                background: #3498db;
                color: white;
            }}
            
            .btn-primary:hover {{
                background: #2980b9;
            }}
            
            .btn-secondary {{
                background: #95a5a6;
                color: white;
            }}
            
            .btn-secondary:hover {{
                background: #7f8c8d;
            }}
            
            .examples-section {{
                margin-bottom: 25px;
            }}
            
            .examples-label {{
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
                display: block;
            }}
            
            .examples-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 10px;
            }}
            
            .example {{
                background: #ecf0f1;
                padding: 12px;
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 0.9em;
                color: #34495e;
                border: 1px solid #bdc3c7;
            }}
            
            .example:hover {{
                background: #d5dbdb;
                border-color: #95a5a6;
            }}
            
            .categories-section {{
                margin-bottom: 30px;
            }}
            
            .categories-label {{
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
                display: block;
            }}
            
            .categories-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 8px;
            }}
            
            .category {{
                background: #34495e;
                color: white;
                padding: 10px 12px;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .result-section {{
                background: #f8f9fa;
                border-radius: 6px;
                padding: 25px;
                margin-top: 20px;
                display: none;
                border: 1px solid #e9ecef;
            }}
            
            .result-section.show {{
                display: block;
                animation: fadeIn 0.3s ease-out;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            
            .result-header {{
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.2em;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 10px;
            }}
            
            .result-item {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 15px;
                padding: 12px;
                background: white;
                border-radius: 4px;
                border-left: 3px solid #3498db;
            }}
            
            .result-label {{
                font-weight: 600;
                color: #2c3e50;
                min-width: 140px;
            }}
            
            .result-value {{
                color: #34495e;
                flex: 1;
                text-align: right;
            }}
            
            .confidence-container {{
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 12px;
            }}
            
            .confidence-bar {{
                width: 120px;
                height: 6px;
                background: #ecf0f1;
                border-radius: 3px;
                overflow: hidden;
            }}
            
            .confidence-fill {{
                height: 100%;
                background: #27ae60;
                transition: width 0.5s ease;
            }}
            
            .loading {{
                text-align: center;
                color: #3498db;
                font-weight: 600;
                padding: 20px;
            }}
            
            .error {{
                background: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
            }}
            
            .safe-status {{
                color: #27ae60;
                font-weight: 600;
            }}
            
            .unsafe-status {{
                color: #e74c3c;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Content Safety Classifier</h1>
                <p>AI-powered content moderation system</p>
            </div>
            
            <div class="status-section">
                <div class="status-item">
                    <span class="status-label">Model Status:</span>
                    <span class="status-value">{model_status}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Model Path:</span>
                    <span class="status-value">{model_info}</span>
                </div>
            </div>
            
            <div class="input-section">
                <label class="input-label">Enter text to analyze</label>
                <textarea class="text-input" id="textInput" placeholder="Enter text content for safety classification..."></textarea>
            </div>
            
            <div class="button-group">
                <button class="btn btn-primary" onclick="classifyText()">Analyze Text</button>
                <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
            </div>
            
            <div class="examples-section">
                <span class="examples-label">Test examples:</span>
                <div class="examples-grid">
                    <div class="example" onclick="setExample('This movie contains explicit nudity and sexual content for adults only')">Adult Content</div>
                    <div class="example" onclick="setExample('People from that religion are evil and should be eliminated')">Hate Speech</div>
                    <div class="example" onclick="setExample('Detailed instructions on how to build bombs and kill innocent people')">Violent Content</div>
                    <div class="example" onclick="setExample('URGENT! You won $1,000,000 click here to claim your prize now!')">Spam Content</div>
                    <div class="example" onclick="setExample('The weather today is sunny with a high of 75 degrees Fahrenheit')">Safe Content</div>
                </div>
            </div>
            
            <div class="categories-section">
                <span class="categories-label">Classification categories:</span>
                <div class="categories-grid">
                    <div class="category">Adult Explicit</div>
                    <div class="category">Hateful Harmful</div>
                    <div class="category">Violent Graphic</div>
                    <div class="category">Spam Promotional</div>
                    <div class="category">Safe Neutral</div>
                </div>
            </div>
            
            <div id="result" class="result-section">
                <div class="result-header">Classification Results</div>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            function setExample(text) {{
                document.getElementById('textInput').value = text;
            }}
            
            function clearAll() {{
                document.getElementById('textInput').value = '';
                document.getElementById('result').classList.remove('show');
            }}
            
            async function classifyText() {{
                const text = document.getElementById('textInput').value.trim();
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (!text) {{
                    alert('Please enter text to analyze.');
                    return;
                }}
                
                resultDiv.classList.add('show');
                resultContent.innerHTML = '<div class="loading">Analyzing content safety...</div>';
                
                try {{
                    const response = await fetch('/classify', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ text: text }})
                    }});
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        const confidence = Math.round(data.confidence * 100);
                        const category = data.category;
                        const categoryDisplay = category.replace(/_/g, ' ').toUpperCase();
                        const isSafe = category === 'safe_neutral';
                        
                        resultContent.innerHTML = `
                            <div class="result-item">
                                <span class="result-label">Input Text</span>
                                <span class="result-value">${{text.length > 80 ? text.substring(0, 80) + '...' : text}}</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label">Predicted Category</span>
                                <span class="result-value">${{categoryDisplay}}</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label">Confidence Level</span>
                                <div class="confidence-container">
                                    <span class="result-value">${{confidence}}%</span>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${{confidence}}%"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="result-item">
                                <span class="result-label">Safety Assessment</span>
                                <span class="result-value ${{isSafe ? 'safe-status' : 'unsafe-status'}}">
                                    ${{isSafe ? 'SAFE CONTENT' : 'UNSAFE CONTENT'}}
                                </span>
                            </div>
                        `;
                    }} else {{
                        throw new Error(data.detail || 'Classification failed');
                    }}
                }} catch (error) {{
                    resultDiv.classList.remove('show');
                    resultDiv.classList.add('error');
                    resultContent.innerHTML = `Error: ${{error.message}}`;
                    setTimeout(() => {{
                        resultDiv.classList.remove('error');
                    }}, 3000);
                }}
            }}
            
            // Allow Enter key to submit (Ctrl+Enter)
            document.getElementById('textInput').addEventListener('keydown', function(event) {{
                if (event.ctrlKey && event.key === 'Enter') {{
                    classifyText();
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
    


@app.get("/test")
async def test():
    if not classifier:
        return {"error": "No model loaded"}
    
    test_cases = [
    "This movie contains explicit nudity and sexual content for adults only",
    "People from that religion are evil and should be eliminated",
    "Detailed instructions on how to build bombs and kill innocent people",
    "URGENT! You won $1,000,000 click here to claim your prize now!",
    "The weather today is sunny with a high of 75 degrees Fahrenheit",
    "Graphic adult material with strong sexual themes and profanity",
    "That race is inferior and doesn't deserve equal rights in society",
    "Step-by-step guide to committing violent acts and terrorism",
    "Make $5000 weekly from home no experience needed sign up now!",
    "I need to go grocery shopping for dinner ingredients tonight",
    "Explicit content with pornographic scenes and adult language",
    "Women belong in the kitchen and shouldn't have professional careers",
    "Methods for self-harm and detailed suicide instructions",
    "Limited time offer buy one get ten free amazing deal!",
    "The library opens at 9 AM and has many educational books",
    "NSFW material with graphic sexual descriptions and nudity",
    "Immigrants are criminals who are destroying our country",
    "Celebrating school shootings and encouraging more violence",
    "Your account has been compromised verify immediately!",
    "Mathematics is an important subject for students to learn"
    ]
    
    results = []
    for text in test_cases:
        try:
            result = classifier(text)
            if isinstance(result, list):
                result = result[0]
            results.append({
                "text": text,
                "prediction": result.get("label", "unknown"),
                "confidence": result.get("score", 0.0)
            })
        except Exception as e:
            results.append({
                "text": text,
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
            })
    
    return {"test_results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
        