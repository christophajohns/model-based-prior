import io
import torch
import uvicorn
import matplotlib.pyplot as plt
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# Store the latest image and rating in memory
latest_image_tensor = None
latest_rating = None

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rate the Image</title>
</head>
<body>
    <h2>Rate the Image</h2>
    <img src="/image" alt="Generated Image" width="256"><br><br>
    <form action="/submit" method="post">
        <label for="rating">Enter rating (0-10):</label>
        <input type="number" id="rating" name="rating" min="0" max="10" step="0.1" required>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_page():
    return HTML_PAGE

@app.get("/image")
def get_image():
    """Serve the current image dynamically without saving it to disk."""
    global latest_image_tensor
    if latest_image_tensor is None:
        return "No image available", 404

    # Convert tensor to an image in memory
    img = latest_image_tensor.cpu().numpy().reshape(16, 16, 3)
    buf = io.BytesIO()
    plt.imsave(buf, img, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

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

def run_server():
    """Start the FastAPI server."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000...")
    run_server()