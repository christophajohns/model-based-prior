# standalone_aesthetics_visualizer.py
import dash
# Old way (pre Dash 2.0): from dash.dependencies import Input, Output, State
from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import torch
import numpy as np
from PIL import Image
import io
import base64
import time
import logging
from torchvision.transforms.functional import resize

# Assuming these imports point to the correct file locations
# If they are in different directories, adjust sys.path or use relative imports carefully
from modelbasedprior.objectives.image_aesthetics.image_aesthetics import (
    ImageAestheticsLoss,
    DEFAULT_PYRAMID_LEVELS,
    DEFAULT_TONE_U_PARAM,
    DEFAULT_TONE_O_PARAM,
)
# IMPORTANT: This import must work and points to the UNMODIFIED image_similarity.py
from modelbasedprior.objectives.image_similarity import generate_image

# --- Helper Functions ---
def numpy_to_base64(im_arr: np.ndarray) -> str:
    """Convert numpy array (H, W, C) or (H, W) uint8 to base64 string"""
    if im_arr.ndim == 3 and im_arr.shape[2] == 1: # Grayscale
        im_arr = im_arr.squeeze(axis=2)
    elif im_arr.ndim == 2: # Grayscale numpy array already (H, W)
        pass # Already correct shape for PIL
    elif im_arr.ndim != 3 or im_arr.shape[2] != 3: # Check for RGB
        raise ValueError(f"Unsupported numpy array shape for image conversion: {im_arr.shape}")

    # Ensure input is uint8 for PIL
    if im_arr.dtype != np.uint8:
        # Try to safely convert if it looks like float [0, 1] or float [0, 255]
        if np.issubdtype(im_arr.dtype, np.floating):
            if im_arr.max() <= 1.0 and im_arr.min() >= 0.0:
                im_arr = (im_arr * 255).astype(np.uint8)
            elif im_arr.max() <= 255.0 and im_arr.min() >= 0.0:
                 im_arr = im_arr.astype(np.uint8)
            else:
                 raise ValueError(f"Cannot safely convert float numpy array with range [{im_arr.min()}, {im_arr.max()}] to uint8")
        else:
             raise ValueError(f"Input array must be uint8 or safely convertible float, got {im_arr.dtype}")


    im_pil = Image.fromarray(im_arr)
    buff = io.BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return "data:image/png;base64," + im_b64

def torch_cwh_to_numpy_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert Torch (C, H, W) uint8 to NumPy (H, W, C) uint8"""
    if tensor.ndim != 3 or tensor.shape[0] not in [1, 3]:
         raise ValueError("Input must be (C, H, W) tensor with 1 or 3 channels.")
    # Ensure tensor is on CPU and detached before converting to numpy
    tensor = tensor.cpu().detach()
    if tensor.shape[0] == 1: # Grayscale
        return tensor.squeeze(0).numpy() # (H, W)
    else: # RGB
        return tensor.permute(1, 2, 0).numpy() # (H, W, C)

def parse_contents(contents):
    """Parse uploaded file contents."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        image = Image.open(io.BytesIO(decoded)).convert('RGB')
        # Convert PIL Image to Torch Tensor (C, H, W) uint8
        img_np = np.array(image) # H, W, C
        img_torch = torch.from_numpy(img_np).permute(2, 0, 1) # C, H, W
        # Ensure it's byte/uint8
        if img_torch.dtype != torch.uint8:
             logging.warning(f"Parsed image tensor was {img_torch.dtype}, converting to uint8.")
             # Assuming the input PIL image was correctly [0, 255]
             img_torch = img_torch.byte()
        return img_torch
    except Exception as e:
        print(f"Error processing image: {e}")
        logging.exception("Image processing failed.")
        return None

# --- Constants and Configuration ---
DEFAULT_IMG_SIZE = 64 # Size used for loss calculation (faster)
MAX_DISPLAY_IMG_SIZE = 256 # Max size for display images
# Use a default random image if none uploaded
DEFAULT_ORIGINAL_IMAGE = (torch.rand(3, MAX_DISPLAY_IMG_SIZE, MAX_DISPLAY_IMG_SIZE) * 255).byte()

# Define bounds from the ImageAestheticsLoss class for sliders
# B, C, S, H
param_names = ["Brightness", "Contrast", "Saturation", "Hue"]
param_ids = ["brightness", "contrast", "saturation", "hue"]
# Use class attributes directly for robustness
try:
    b_bounds = ImageAestheticsLoss._brightness_bounds
    c_bounds = ImageAestheticsLoss._contrast_bounds
    s_bounds = ImageAestheticsLoss._saturation_bounds
    h_bounds = ImageAestheticsLoss._hue_bounds
    default_bounds = [b_bounds, c_bounds, s_bounds, h_bounds]
except AttributeError:
    logging.warning("Could not access bounds from ImageAestheticsLoss, using fallback defaults.")
    default_bounds = [(0.8, 1.25), (0.8, 1.25), (0.8, 1.25), (-0.25, 0.25)]

default_values = [1.0, 1.0, 1.0, 0.0] # Neutral values

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], suppress_callback_exceptions=True)
server = app.server # For deployment (e.g., Heroku)

# --- Layout Definition ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Interactive Image Aesthetics Explorer"), width=12)),
    dbc.Row([
        # --- Left Column: Controls ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("1. Upload Image"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                        },
                        multiple=False # Allow only single image upload
                    ),
                     html.Div(id='output-image-upload-info', style={'textAlign': 'center', 'fontSize': 'small'}),
                ])
            ]),
            html.Hr(),
            dbc.Card([
                dbc.CardHeader("2. Adjust Parameters"),
                dbc.CardBody([
                    # Dynamically create sliders for B, C, S, H
                    *[dbc.Row([
                        dbc.Col(html.Label(name), width=3),
                        dbc.Col(dcc.Slider(
                            id=f'slider-{p_id}', min=b[0], max=b[1], step=0.01, value=val,
                            marks={b[0]: f"{b[0]:.2f}", val: f"{val:.2f}", b[1]: f"{b[1]:.2f}"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ), width=7),
                        dbc.Col(dcc.Input(
                            id=f'input-{p_id}', type='number', value=val,
                            min=b[0], max=b[1], step=0.01, style={'width': '100%'}
                        ), width=2)
                    ], className="mb-3 align-items-center")
                    for name, p_id, b, val in zip(param_names, param_ids, default_bounds, default_values)]
                ])
            ]),
            html.Hr(),
             dbc.Card([
                dbc.CardHeader("3. Loss Calculation Settings"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label("Pyramid Levels (K)"), width=6),
                        dbc.Col(dcc.Slider(
                            id='slider-k-levels', min=2, max=10, step=1, value=DEFAULT_PYRAMID_LEVELS,
                            marks={i: str(i) for i in range(2, 11)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ), width=6)
                    ], className="mb-2 align-items-center"),
                     dbc.Row([
                        dbc.Col(html.Label("Tone Under. (u)"), width=6),
                        dbc.Col(dcc.Input(
                            id='input-tone-u', type='number', value=DEFAULT_TONE_U_PARAM,
                             min=0.01, max=0.5, step=0.01, style={'width': '100%'}
                        ), width=6)
                    ], className="mb-2 align-items-center"),
                     dbc.Row([
                        dbc.Col(html.Label("Tone Over. (o)"), width=6),
                        dbc.Col(dcc.Input(
                            id='input-tone-o', type='number', value=DEFAULT_TONE_O_PARAM,
                             min=0.01, max=0.5, step=0.01, style={'width': '100%'}
                        ), width=6)
                    ], className="mb-2 align-items-center"),
                    dbc.Row([
                        dbc.Col(html.Label("Internal Size (px)"), width=6),
                        dbc.Col(dcc.Slider(
                            id='slider-internal-size', min=32, max=128, step=16, value=DEFAULT_IMG_SIZE,
                            marks={i: str(i) for i in [32, 64, 96, 128]},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ), width=6)
                    ], className="mb-2 align-items-center"),
                ])
            ]),
             html.Br(),
             dbc.Button("Recalculate Score & Metrics", id="recalculate-button", color="primary", className="w-100"),
             html.Div(id='calculation-status', style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': 'small'}),

        ], width=12, lg=4),

        # --- Right Column: Outputs ---
        dbc.Col([
            dbc.Row([
                dbc.Col(html.H4("Original Image"), width=6, className="text-center"),
                dbc.Col(html.H4("Modified Image"), width=6, className="text-center"),
            ]),
            dbc.Row([
                 dbc.Col(html.Img(id='display-original-image', style={'max-width': '100%', 'height': 'auto'}), width=6),
                 dbc.Col(html.Img(id='display-modified-image', style={'max-width': '100%', 'height': 'auto'}), width=6),
            ]),
            html.Hr(),
             dbc.Row([
                dbc.Col(html.H4("Aesthetic Score & Metrics"), width=12, className="text-center"),
            ]),
             dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(id='score-display'), color="info", outline=True), width=12, md=4),
                dbc.Col(dbc.Card(dbc.CardBody(id='metrics-display'), color="secondary", outline=True), width=12, md=8),
             ]),
             html.Hr(),
             dbc.Row([
                dbc.Col(html.H4("Parameter Sensitivity (1D Slices)"), width=12, className="text-center"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Parameter to Vary:"),
                    dcc.Dropdown(
                        id='slice-param-selector',
                        options=[{'label': name, 'value': p_id} for name, p_id in zip(param_names, param_ids)],
                        value=param_ids[0], # Default to Brightness
                        clearable=False
                    ),
                     dbc.Button("Generate 1D Slice Plot", id="plot-slice-button", color="success", className="w-100 mt-2"),
                ], width=12, md=4),
                dbc.Col(dcc.Loading(
                    id="loading-slice-plot",
                    type="circle",
                    children=[dcc.Graph(id='slice-plot-graph')] # Wrap initial graph in loading
                 ), width=12, md=8),
            ]),


        ], width=12, lg=8),
    ]),

    # Hidden storage for the original image tensor (as base64 string for serialization)
    dcc.Store(id='original-image-store'),
    # Hidden storage for computed metrics (optional, could recompute)
    dcc.Store(id='computed-metrics-store'),
     # Hidden storage for loss config
     dcc.Store(id='loss-config-store'),

], fluid=True)

# --- Callbacks ---

# Update sliders from number inputs and vice-versa
for p_id in param_ids:
    @app.callback(
        Output(f'slider-{p_id}', 'value', allow_duplicate=True),
        Output(f'input-{p_id}', 'value', allow_duplicate=True),
        Input(f'slider-{p_id}', 'value'),
        Input(f'input-{p_id}', 'value'),
        prevent_initial_call=True # Important to prevent loops on startup
    )
    def sync_slider_input(slider_val, input_val):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == f'slider-{p_id}':
            # Slider changed, update input only if different (with tolerance)
            if slider_val is not None and (input_val is None or not np.isclose(slider_val, input_val)):
                 return no_update, slider_val
            else:
                 return no_update, no_update
        elif trigger_id == f'input-{p_id}':
            # Input changed, update slider only if different (with tolerance)
            if input_val is not None and (slider_val is None or not np.isclose(input_val, slider_val)):
                 return input_val, no_update
            else:
                 return no_update, no_update
        else:
            return no_update, no_update

# Store uploaded image
@app.callback(
    Output('original-image-store', 'data'),
    Output('output-image-upload-info', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def store_uploaded_image(contents):
    if contents:
        img_torch = parse_contents(contents)
        if img_torch is not None and img_torch.dtype == torch.uint8: # Check type
            # Store as base64 for JSON serialization in dcc.Store
            # Convert tensor (C, H, W) uint8 to numpy (H, W, C) uint8 for PIL
            img_np_hwc = torch_cwh_to_numpy_hwc(img_torch)
            img_pil = Image.fromarray(img_np_hwc)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            info = f"Loaded image ({img_torch.shape[2]}x{img_torch.shape[1]})"
            # Store image string and shape
            return {'image_b64': img_str, 'shape': list(img_torch.shape)}, info
        elif img_torch is None:
            return no_update, "Failed to process image."
        else:
            # This case should ideally not happen if parse_contents ensures uint8
            return no_update, f"Image parsed but had unexpected type: {img_torch.dtype}"
    return no_update, "Upload an image."

# Store loss configuration
@app.callback(
    Output('loss-config-store', 'data'),
    Input('slider-k-levels', 'value'),
    Input('input-tone-u', 'value'),
    Input('input-tone-o', 'value'),
    Input('slider-internal-size', 'value')
)
def store_loss_config(k, u, o, size):
    # Basic validation
    k = int(k) if k else DEFAULT_PYRAMID_LEVELS
    u = float(u) if u else DEFAULT_TONE_U_PARAM
    o = float(o) if o else DEFAULT_TONE_O_PARAM
    size = int(size) if size else DEFAULT_IMG_SIZE
    return {'k': k, 'u': u, 'o': o, 'size': size}


# Main callback to update modified image, score, and metrics
@app.callback(
    Output('display-original-image', 'src'),
    Output('display-modified-image', 'src'),
    Output('score-display', 'children'),
    Output('metrics-display', 'children'),
    Output('computed-metrics-store', 'data'),
    Output('calculation-status', 'children'),
    Input('original-image-store', 'data'),
    Input('loss-config-store', 'data'),
    Input('recalculate-button', 'n_clicks'),
    State('slider-brightness', 'value'),
    State('slider-contrast', 'value'),
    State('slider-saturation', 'value'),
    State('slider-hue', 'value'),
    prevent_initial_call=True # Prevent firing on startup
)
def update_main_output(stored_image_data, loss_config, n_clicks, brightness, contrast, saturation, hue):
    ctx = dash.callback_context

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial load'
    logging.info(f"update_main_output triggered by: {trigger_id}")

    # Handle initial load or cases where essential data is missing
    if not stored_image_data or not loss_config:
         logging.info("Initial load or missing data, showing defaults.")
         default_img_src = numpy_to_base64(torch_cwh_to_numpy_hwc(DEFAULT_ORIGINAL_IMAGE))
         return (
             default_img_src, # Original
             default_img_src, # Modified
             [html.H5("Aesthetic Score", className="card-title"), html.P("N/A", style={'fontSize': '1.2em'})],
             [html.H5("Component Metrics", className="card-title"), html.P("Upload an image and adjust parameters.", style={'fontSize': 'small'})],
             {}, # Empty metrics store
             "Please upload an image."
         )

    # --- 1. Decode Stored Image ---
    try:
        img_str = stored_image_data['image_b64']
        # Decode base64 back to PIL image, then to numpy HWC, then to torch CWH uint8
        img_pil = Image.open(io.BytesIO(base64.b64decode(img_str))).convert('RGB')
        img_np_hwc = np.array(img_pil)
        img_torch_orig_uint8 = torch.from_numpy(img_np_hwc).permute(2, 0, 1).byte() # C, H, W uint8
        logging.info(f"Decoded original image: {img_torch_orig_uint8.shape}, dtype={img_torch_orig_uint8.dtype}")
        assert img_torch_orig_uint8.dtype == torch.uint8 # Sanity check

    except Exception as e:
        logging.exception("Error decoding stored image:")
        error_msg = f"Error loading stored image: {e}"
        # Show default images but indicate error
        default_img_src = numpy_to_base64(torch_cwh_to_numpy_hwc(DEFAULT_ORIGINAL_IMAGE))
        return default_img_src, default_img_src, [html.H5("Error", className="card-title text-danger"), html.P(error_msg)], [html.H5("Error", className="card-title text-danger")], {}, f"Error: {error_msg}"

    # --- 2. Prepare Images ---
    start_time = time.time()
    status_update = ["Processing..."]

    # -- Display Image Generation --
    # Resize original uint8 image for display if needed
    h_orig, w_orig = img_torch_orig_uint8.shape[1:]
    scale = min(1.0, MAX_DISPLAY_IMG_SIZE / max(h_orig, w_orig, 1))
    if scale < 1.0:
        display_h, display_w = int(h_orig * scale), int(w_orig * scale)
        img_torch_display_orig_uint8 = resize(img_torch_orig_uint8, [display_h, display_w], antialias=True)
    else:
        img_torch_display_orig_uint8 = img_torch_orig_uint8

    original_src = numpy_to_base64(torch_cwh_to_numpy_hwc(img_torch_display_orig_uint8))

    # Generate modified image FOR DISPLAY
    # Pass the display-sized UINT8 original image to generate_image
    X_params = torch.tensor([[[brightness, contrast, saturation, hue]]], dtype=torch.float32) # Shape (1, 1, 4)
    logging.info(f"Generating display image with params: {X_params}")
    # generate_image expects uint8 [0, 255], returns float tensor likely [0, 255]
    modified_img_display_float = generate_image(X_params, img_torch_display_orig_uint8).squeeze() # Squeeze n=1, q=1 -> (C, H, W) float

    # Convert float output [0, 255] to uint8 [0, 255] for display encoding
    # Clamp to handle potential small out-of-bounds values from transforms
    modified_img_display_uint8 = modified_img_display_float.clamp(0, 255).byte()
    modified_src = numpy_to_base64(torch_cwh_to_numpy_hwc(modified_img_display_uint8))

    # -- Internal Image Generation (for loss calculation) --
    internal_size = loss_config.get('size', DEFAULT_IMG_SIZE)
    # Resize the original UINT8 image to the internal calculation size
    img_torch_internal_uint8 = resize(img_torch_orig_uint8, [internal_size, internal_size], antialias=True)
    logging.info(f"Internal image size: {img_torch_internal_uint8.shape}")

    # Generate internal modified image using the internal-sized UINT8 original
    # X_params is already defined as (1, 1, 4)
    logging.info(f"Generating internal image with params: {X_params}")
    # generate_image expects uint8 [0, 255], returns float tensor likely [0, 255]
    modified_img_internal_float = generate_image(X_params, img_torch_internal_uint8).squeeze() # (C, H, W) float

    status_update.append(f"Image Gen: {time.time()-start_time:.2f}s")

    # --- 3. Calculate Loss and Metrics ---
    metrics_calc_start_time = time.time()
    try:
        # Instantiate loss function with the *original internal* uint8 image
        # Note: ImageAestheticsLoss constructor doesn't actually use the image pixels,
        # but passing the correct size might be relevant if it did.
        # The actual calculation happens in _compute_all_metrics using the *modified* image.
        loss_func = ImageAestheticsLoss(
            original_image=img_torch_internal_uint8, # Pass uint8 internal original
            k_levels=loss_config.get('k', DEFAULT_PYRAMID_LEVELS),
            tone_u=loss_config.get('u', DEFAULT_TONE_U_PARAM),
            tone_o=loss_config.get('o', DEFAULT_TONE_O_PARAM),
            negate=True # We want the score (higher is better)
        )

        # Prepare the *modified internal float* image for _compute_all_metrics
        # _compute_all_metrics expects a BATCH of FLOAT tensors in range [0, 255]
        # It internally divides by 255.0.
        img_batch_float_0_255 = modified_img_internal_float.unsqueeze(0) # Add batch dim -> (1, C, H, W) float [0, 255]
        logging.info(f"Input to _compute_all_metrics: shape={img_batch_float_0_255.shape}, dtype={img_batch_float_0_255.dtype}, range=[{img_batch_float_0_255.min():.2f}, {img_batch_float_0_255.max():.2f}]")

        # Compute raw metrics using the loss function's internal method
        sh, de, cl, to, co = loss_func._compute_all_metrics(img_batch_float_0_255) # Pass float [0, 255]

        # Combine metrics for the final score (as done in ImageAestheticsLoss.evaluate_true)
        other_metrics = torch.stack([de, cl, to, co], dim=1) # (1, 4)
        mean_others = other_metrics.mean(dim=1) # (1,)
        score = sh * mean_others # (1,)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        final_score = score.item() # Get scalar value

        metrics_dict = {
            "Sharpness": sh.item(), "Depth": de.item(),
            "Clarity": cl.item(), "Tone": to.item(), "Colorfulness": co.item()
        }
        status_update.append(f"Metrics Calc: {time.time()-metrics_calc_start_time:.2f}s")
        logging.info(f"Calculated score: {final_score:.4f}, Metrics: {metrics_dict}")

    except Exception as e:
        logging.exception("Error during score calculation")
        final_score = "Error"
        metrics_dict = {"Error": str(e)}
        status_update.append("Metrics Calc: Error")


    # --- 4. Format Output ---
    if isinstance(final_score, float):
        score_card_content = [
            html.H5("Aesthetic Score", className="card-title"),
            html.P(f"{final_score:.4f}", style={'fontSize': '1.5em', 'fontWeight': 'bold'}),
            html.P("(Higher is better)", style={'fontSize': 'small'})
        ]
    else:
         score_card_content = [
            html.H5("Aesthetic Score", className="card-title"),
            html.P("Calculation Error", style={'fontSize': '1.2em', 'fontWeight': 'bold', 'color': 'red'})
        ]

    metrics_card_content = [html.H5("Component Metrics", className="card-title")]
    if "Error" not in metrics_dict:
        metrics_card_content.extend([
            html.P(f"{name}: {value:.4f}", style={'margin': '0.1rem 0'}) for name, value in metrics_dict.items()
        ])
    else:
         metrics_card_content.append(html.P(f"Error: {metrics_dict['Error']}", style={'color': 'red'}))

    total_time = time.time() - start_time
    status_update.append(f"Total: {total_time:.2f}s")
    final_status = " | ".join(status_update)

    return original_src, modified_src, score_card_content, metrics_card_content, metrics_dict, final_status


# Callback to generate 1D slice plot
@app.callback(
    # Output('slice-plot-graph', 'figure'), # Original way, causes flicker
    Output("loading-slice-plot", "children"), # Update the content of the loading component
    Input('plot-slice-button', 'n_clicks'),
    State('slice-param-selector', 'value'),
    State('original-image-store', 'data'),
    State('loss-config-store', 'data'),
    State('slider-brightness', 'value'),
    State('slider-contrast', 'value'),
    State('slider-saturation', 'value'),
    State('slider-hue', 'value'),
    prevent_initial_call=True,
)
def generate_slice_plot(n_clicks, slice_param_id, stored_image_data, loss_config, br, co, sa, hu):
    logging.info(f"generate_slice_plot triggered for param: {slice_param_id}")
    # Placeholder figure while calculating or if inputs missing
    placeholder_fig = go.Figure().update_layout(
        title="Select Parameter and Click 'Generate Plot'",
        xaxis_title="Parameter Value",
        yaxis_title="Aesthetic Score",
         xaxis={'range': [0,1]}, # Default range to avoid empty plot look
         yaxis={'range': [0,1]}
    )

    if not n_clicks or not stored_image_data or not loss_config or not slice_param_id:
        logging.warning("Slice plot called without necessary inputs.")
        return [dcc.Graph(figure=placeholder_fig)] # Return placeholder inside list for children

    # --- 1. Get Base Image and Config ---
    try:
        img_str = stored_image_data['image_b64']
        # Decode base64 back to PIL image, then to numpy HWC, then to torch CWH uint8
        img_pil = Image.open(io.BytesIO(base64.b64decode(img_str))).convert('RGB')
        img_np_hwc = np.array(img_pil)
        img_torch_orig_uint8 = torch.from_numpy(img_np_hwc).permute(2, 0, 1).byte() # C, H, W uint8
        assert img_torch_orig_uint8.dtype == torch.uint8 # Sanity check
    except Exception as e:
        logging.exception("Error decoding stored image for slice plot:")
        error_fig = go.Figure().update_layout(title=f"Error loading image: {e}")
        return [dcc.Graph(figure=error_fig)] # Return error plot inside list

    internal_size = loss_config.get('size', DEFAULT_IMG_SIZE)
    # Resize the original UINT8 image
    img_torch_internal_uint8 = resize(img_torch_orig_uint8, [internal_size, internal_size], antialias=True)
    logging.info(f"Slice plot using internal image: {img_torch_internal_uint8.shape}, dtype={img_torch_internal_uint8.dtype}")

    # Instantiate loss function (needs dummy uint8 image in constructor)
    loss_func = ImageAestheticsLoss(
        original_image=img_torch_internal_uint8, # Pass uint8 internal original
        k_levels=loss_config.get('k', DEFAULT_PYRAMID_LEVELS),
        tone_u=loss_config.get('u', DEFAULT_TONE_U_PARAM),
        tone_o=loss_config.get('o', DEFAULT_TONE_O_PARAM),
        negate=True # We want the score
    )

    # --- 2. Define Parameter Range and Fixed Values ---
    current_params = {'brightness': br, 'contrast': co, 'saturation': sa, 'hue': hu}
    try:
        param_idx = param_ids.index(slice_param_id)
    except ValueError:
         logging.error(f"Invalid slice parameter ID: {slice_param_id}")
         error_fig = go.Figure().update_layout(title=f"Invalid parameter selected: {slice_param_id}")
         return [dcc.Graph(figure=error_fig)]

    bounds = default_bounds[param_idx]
    param_range = torch.linspace(bounds[0], bounds[1], steps=25) # 25 steps for the plot

    scores = []
    param_values = []
    fixed_params_desc = ", ".join([f"{name[:2]}={current_params[pid]:.2f}" for pid, name in zip(param_ids, param_names) if pid != slice_param_id])

    # --- 3. Iterate and Calculate Scores ---
    start_plot_calc = time.time()
    for val in param_range:
        # Construct the parameter tensor (1, 1, 4) for this step
        X_eval = torch.tensor([[
            val.item() if slice_param_id == 'brightness' else current_params['brightness'],
            val.item() if slice_param_id == 'contrast' else current_params['contrast'],
            val.item() if slice_param_id == 'saturation' else current_params['saturation'],
            val.item() if slice_param_id == 'hue' else current_params['hue'],
        ]], dtype=torch.float32).unsqueeze(0) # Shape (1, 1, 4)

        # Generate the single modified image using the internal UINT8 original
        # generate_image expects uint8 [0, 255], returns float tensor likely [0, 255]
        modified_img_internal_float = generate_image(X_eval, img_torch_internal_uint8).squeeze() # (C, H, W) float [0, 255]

        try:
            # Compute metrics and score using the internal float modified image
            # Prepare batch: float [0, 255] -> add batch dim -> (1, C, H, W) float [0, 255]
            img_batch_float_0_255 = modified_img_internal_float.unsqueeze(0)

            sh, de, cl, to, co_metric = loss_func._compute_all_metrics(img_batch_float_0_255) # Pass float [0, 255]

            other_metrics = torch.stack([de, cl, to, co_metric], dim=1)
            mean_others = other_metrics.mean(dim=1)
            score = sh * mean_others
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

            scores.append(score.item())
            param_values.append(val.item())

        except Exception as e:
            # Log error but continue calculation for other points
            logging.warning(f"Warning: Error calculating score for {slice_param_id}={val}: {e}", exc_info=False) # Keep log cleaner
            scores.append(np.nan) # Add NaN if calculation fails
            param_values.append(val.item())

    logging.info(f"Slice plot calculation took {time.time() - start_plot_calc:.2f}s")

    # --- 4. Create Plot ---
    fig = go.Figure()
    # Filter out NaNs for plotting lines correctly, but keep original param_values for axis
    valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
    if valid_indices:
         fig.add_trace(go.Scatter(
            x=np.array(param_values)[valid_indices],
            y=np.array(scores)[valid_indices],
            mode='lines+markers',
            name='Aesthetic Score'
        ))
    else:
         # Add dummy trace if all failed, so plot shows ranges
         fig.add_trace(go.Scatter(x=param_values, y=[0]*len(param_values), mode='markers', marker=dict(opacity=0), name='Calculation Failed'))


    # Add vertical line for current slider value
    current_val = current_params[slice_param_id]
    fig.add_vline(x=current_val, line_width=2, line_dash="dash", line_color="red",
                  annotation_text="Current", annotation_position="top left")

    param_name = param_names[param_idx]
    fig.update_layout(
        title=f"Score vs. {param_name}<br>(Fixed: {fixed_params_desc})",
        xaxis_title=f"{param_name} Value",
        yaxis_title="Aesthetic Score",
        margin=dict(l=40, r=20, t=60, b=40), # Adjusted margins
        xaxis = {'range': [bounds[0], bounds[1]]}, # Set x-axis range explicitly
        # Optionally set y-axis range based on calculated scores
        # yaxis = {'range': [min(s for s in scores if not np.isnan(s)) - buffer, max(s for s in scores if not np.isnan(s)) + buffer]} if valid_indices else {'range': [0, 1]}
    )

    # Return the Graph component wrapped in a list for the Loading component's children
    return [dcc.Graph(figure=fig)]


# --- Run the App ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Dash app running on http://127.0.0.1:8050/")
    # Set debug=False for production or if experiencing reload issues
    # Set debug=True for development (enables hot-reloading and error pages)
    app.run(debug=True, port=8050)