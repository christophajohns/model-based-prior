import io
import torch
import uvicorn
import matplotlib.pyplot as plt
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from jinja2 import Template

app = FastAPI()

# Store the latest image and rating in memory
latest_image_tensor: torch.Tensor | None = None
latest_rating: float | None = None
target_image_tensor: torch.Tensor | None = None

def generate_html(has_target_image: bool):
    template = Template("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Rating</title>
        <style>
            :root {
                --bg-color: #f4f4f8;
                --text-color: #333;
                --accent-color: #3498db;
                --card-bg: white;
                --border-radius: 8px;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                line-height: 1.6;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 1rem;
            }
            
            .container {
                background-color: var(--card-bg);
                border-radius: var(--border-radius);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 2rem;
                width: 100%;
                max-width: 700px;
                text-align: center;
            }
            
            .image-container {
                display: flex;
                justify-content: center;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
                flex-wrap: wrap;
            }
            
            figure {
                background-color: var(--bg-color);
                border-radius: var(--border-radius);
                padding: 1rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
            }
            
            img {
                max-width: 256px;
                height: auto;
                border-radius: var(--border-radius);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            figcaption {
                color: var(--text-color);
                opacity: 0.7;
            }
            
            form {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1rem;
            }
            
            label {
                font-weight: 500;
            }
            
            input[type="number"] {
                width: 100px;
                padding: 0.5rem;
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            
            button {
                background-color: var(--accent-color);
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            
            button:hover {
                background-color: #2980b9;
            }
            
            h2 {
                margin-bottom: 1.5rem;
                color: var(--text-color);
            }
        </style>
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelector('form').addEventListener('submit', function(e) {
                e.preventDefault();
                const rating = document.getElementById('rating').value;
                
                fetch('/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `rating=${rating}`
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        window.location.reload();
                    }
                });
            });
        });
        </script>
    </head>
    <body>
        <div class="container">
            <h2>Rate the Image</h2>
            <div class="image-container">
                <figure>
                    <img src="/image" alt="Generated Image" width="256" />
                    <figcaption>Generated image</figcaption>
                </figure>
                {% if has_target_image %}
                <figure>
                    <img src="/target-image" alt="Reference Image" width="256" />
                    <figcaption>Target image</figcaption>
                </figure>
                {% endif %}
            </div>
            <form method="post">
                <label for="rating">Enter rating (0-10):</label>
                <input type="number" id="rating" name="rating" min="0" max="10" step="0.1" required>
                <button type="submit">Submit Rating</button>
            </form>
        </div>
    </body>
    </html>
    """)
    
    return template.render(has_target_image=has_target_image)

def serve_image(image_tensor: torch.Tensor | None):
    if image_tensor is None:
        return "No image available", 404

    # Handle different tensor shapes
    img_array = image_tensor.cpu().numpy()
    if img_array.ndim == 2:  # Grayscale image case
        cmap = "gray"
    elif img_array.ndim == 3 and img_array.shape[-1] in {1, 3, 4}:  
        # Single-channel, RGB, or RGBA
        cmap = None
    elif img_array.ndim == 3 and img_array.shape[0] in {1, 3, 4}:  
        # (C, H, W) format, so we transpose to (H, W, C)
        img_array = img_array.transpose(1, 2, 0)
        cmap = None
    else:
        return "Unsupported image format", 400
    
    # Convert the image array to an in-memory PNG file
    buf = io.BytesIO()
    plt.imsave(buf, img_array, cmap=cmap, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
def serve_page():
    return generate_html(has_target_image=target_image_tensor is not None)

@app.get("/image")
def get_image():
    """Serve the current image dynamically without saving it to disk."""
    global latest_image_tensor
    return serve_image(image_tensor=latest_image_tensor)

@app.get("/target-image")
def get_target_image():
    """Serve the target image dynamically without saving it to disk."""
    global target_image_tensor
    return serve_image(image_tensor=target_image_tensor)

@app.post("/submit")
def submit_rating(rating: float = Form(...)):
    """Receive the user rating and store it in a shared variable."""
    global latest_rating
    latest_rating = rating  # Store rating in memory
    return {"status": "success"}  # Send simple success response

def set_latest_image(tensor: torch.Tensor):
    """Set the latest image tensor to be displayed on the web UI."""
    global latest_image_tensor
    latest_image_tensor = tensor

def get_latest_rating():
    """Retrieve the latest rating (used by WebHumanEvaluatorRenderer)."""
    global latest_rating
    return latest_rating

def reset_latest_rating():
    """Reset the latest rating to ensure each new image gets fresh input."""
    global latest_rating
    latest_rating = None

def set_target_image(tensor: torch.Tensor):
    """Set the target image."""
    global target_image_tensor
    target_image_tensor = tensor

def run_server():
    """Start the FastAPI server."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000...")
    run_server()