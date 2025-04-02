import dash
# Old way (pre Dash 2.0): from dash.dependencies import Input, Output, State
from dash import Input, Output, State, dcc, html, no_update, ctx # Import ctx
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
LIVE_UPDATE_DELAY_MS = 250 # Minimum milliseconds between live image updates

# Define bounds from the ImageAestheticsLoss class for sliders
# B, C, S, H
param_names = ["Brightness", "Contrast", "Saturation", "Hue"]
param_ids = ["brightness", "contrast", "saturation", "hue"]
param_options = [{'label': name, 'value': p_id} for name, p_id in zip(param_names, param_ids)]
# Use class attributes directly for robustness
try:
    b_bounds = ImageAestheticsLoss._brightness_bounds
    c_bounds = ImageAestheticsLoss._contrast_bounds
    s_bounds = ImageAestheticsLoss._saturation_bounds
    h_bounds = ImageAestheticsLoss._hue_bounds
    default_bounds = [b_bounds, c_bounds, s_bounds, h_bounds]
    param_bounds_map = dict(zip(param_ids, default_bounds))
except AttributeError:
    logging.warning("Could not access bounds from ImageAestheticsLoss, using fallback defaults.")
    default_bounds = [(0.8, 1.25), (0.8, 1.25), (0.8, 1.25), (-0.25, 0.25)]
    param_bounds_map = dict(zip(param_ids, default_bounds))


default_values = [1.0, 1.0, 1.0, 0.0] # Neutral values
param_defaults_map = dict(zip(param_ids, default_values))

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
                    # Live Update Toggle
                    dbc.Row([
                        dbc.Col(dbc.Label("Live Image Update:", html_for="live-update-toggle"), width=8),
                        dbc.Col(dbc.Switch(id="live-update-toggle", value=True, label="On/Off"), width=4),
                    ], className="mb-3 align-items-center"),
                    html.Hr(),
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
                        options=param_options,
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

            # --- 4D Landscape Visualization ---
            html.Hr(),
            dbc.Row([
                dbc.Col(html.H4("Objective Landscape (2D Slices / Heatmap)"), width=12, className="text-center"),
            ]),
            dbc.Row([
                # Controls Column
                dbc.Col([
                    dbc.Label("X-Axis Parameter:"),
                    dcc.Dropdown(id='heatmap-x-axis', options=param_options, value=param_ids[0], clearable=False),
                    html.Br(),
                    dbc.Label("Y-Axis Parameter:"),
                    dcc.Dropdown(id='heatmap-y-axis', options=param_options, value=param_ids[1], clearable=False),
                    html.Br(),

                    # --- START: ADDED RESOLUTION SLIDER ---
                    dbc.Label("Heatmap Resolution (Grid Size):"),
                    dcc.Slider(
                        id='heatmap-resolution-slider',
                        min=5,        # Minimum sensible grid size
                        max=50,       # Maximum sensible grid size (can increase, but >50 gets slow)
                        step=5,       # Step size
                        value=20,     # Default value (matches old constant)
                        marks={i: str(i) for i in range(5, 51, 5)}, # Marks for clarity
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(), # Add some spacing
                    
                    # Dynamically updated fixed value inputs
                    dbc.Row([
                         dbc.Col(dbc.Label("Fixed Param 1:", id='fixed-param-1-label'), width=6),
                         dbc.Col(dcc.Input(id='fixed-param-1-input', type='number', style={'width': '100%'}), width=6),
                    ], className="mb-2 align-items-center"),
                     dbc.Row([
                         dbc.Col(dbc.Label("Fixed Param 2:", id='fixed-param-2-label'), width=6),
                         dbc.Col(dcc.Input(id='fixed-param-2-input', type='number', style={'width': '100%'}), width=6),
                    ], className="mb-2 align-items-center"),
                    html.Br(),
                    dbc.Button("Generate 2D Heatmap", id="plot-heatmap-button", color="warning", className="w-100"),
                    html.Div(id='heatmap-calculation-status', style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': 'small'}),
                ], width=12, md=4),
                # Plot Column
                dbc.Col(dcc.Loading(
                    id="loading-heatmap-plot",
                    type="circle",
                    children=[dcc.Graph(id='heatmap-plot-graph')] # Wrap initial graph in loading
                 ), width=12, md=8),
            ]),

            # --- 4D Parallel Coordinates Plot Section ---
            html.Hr(),
            dbc.Row([
                dbc.Col(html.H4("4D Exploration: Parallel Coordinates Plot"), width=12, className="text-center"),
            ]),
            dbc.Row([
                # PCP Controls Column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("PCP Settings"),
                        dbc.CardBody([
                            dbc.Label("Points per Dimension:", html_for="pcp-points-slider"),
                            dcc.Slider(
                                id='pcp-points-slider',
                                min=2,
                                max=7, # Max 7^4 = 2401 points. Higher gets very slow.
                                step=1,
                                value=4, # Default: 4^4 = 256 points
                                marks={i: str(i) for i in range(2, 8)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.P(id='pcp-total-points-info', style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': 'small'}),
                            html.Br(),
                            dbc.Button("Generate Parallel Coordinates Plot", id="generate-pcp-button", color="info", className="w-100"),
                            html.Div(id='pcp-calculation-status', style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': 'small'}),
                        ])
                    ])
                ], width=12, md=4),

                # PCP Plot Column
                dbc.Col(dcc.Loading(
                    id="loading-pcp-plot",
                    type="circle",
                    children=[dcc.Graph(id='pcp-graph')] # Wrap initial graph in loading
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
    # Hidden store for fixed parameter IDs (to help heatmap callback)
    dcc.Store(id='fixed-param-ids-store', data={'p1': None, 'p2': None}),


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
    def sync_slider_input(slider_val, input_val, param_id=p_id): # Need to capture p_id
        # Use dash.ctx instead of callback_context
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return no_update, no_update

        if triggered_id == f'slider-{param_id}':
            # Slider changed, update input only if different (with tolerance)
            if slider_val is not None and (input_val is None or not np.isclose(slider_val, input_val)):
                 return no_update, slider_val
            else:
                 return no_update, no_update
        elif triggered_id == f'input-{param_id}':
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
    # --- Inputs that ALWAYS trigger update ---
    Input('original-image-store', 'data'),
    Input('loss-config-store', 'data'),
    Input('recalculate-button', 'n_clicks'),
    # --- Inputs that trigger update ONLY IF Live Update is ON ---
    Input('slider-brightness', 'value'),
    Input('slider-contrast', 'value'),
    Input('slider-saturation', 'value'),
    Input('slider-hue', 'value'),
    # --- State needed for the logic ---
    State('live-update-toggle', 'value'),
    # Note: Slider values are now passed directly as arguments from Inputs
    prevent_initial_call=True # Prevent firing on startup
)
def update_main_output(
    stored_image_data, loss_config, n_clicks, # Always trigger inputs
    brightness, contrast, saturation, hue,    # Slider inputs
    live_update_on                            # State of the toggle
):
    trigger_id = ctx.triggered_id
    logging.info(f"update_main_output triggered by: {trigger_id} | Live Update State: {live_update_on}") # Log toggle state

    # --- Live Update Logic ---
    is_slider_trigger = trigger_id and trigger_id.startswith('slider-')

    if is_slider_trigger and not live_update_on:
        logging.info("Live update OFF, slider trigger ignored. No recalculation.")
        # Return no_update for all outputs (must match the number of Outputs)
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    logging.info("Proceeding with recalculation...") # Add log to confirm calculation proceeds

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
        sh_raw, de_raw, cl_raw, to_raw, co_raw = loss_func._compute_all_metrics(img_batch_float_0_255) # Pass float [0, 255]
        sh, de, cl, to, co = loss_func._normalize_metrics(sh_raw, de_raw, cl_raw, to_raw, co_raw)

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
        param_name_map = dict(zip(param_ids, param_names))
        param_name = param_name_map[slice_param_id]
    except ValueError:
         logging.error(f"Invalid slice parameter ID: {slice_param_id}")
         error_fig = go.Figure().update_layout(title=f"Invalid parameter selected: {slice_param_id}")
         return [dcc.Graph(figure=error_fig)]

    bounds = param_bounds_map[slice_param_id]
    param_range = torch.linspace(bounds[0], bounds[1], steps=25) # 25 steps for the plot

    scores = []
    param_values = []
    fixed_params_desc = ", ".join([f"{name[:2]}={current_params[pid]:.2f}" for pid, name in zip(param_ids, param_names) if pid != slice_param_id])

    # --- 3. Iterate and Calculate Scores ---
    start_plot_calc = time.time()
    for val in param_range:
        # Construct the parameter tensor (1, 1, 4) for this step
        X_eval_list = []
        for pid in param_ids:
            if pid == slice_param_id:
                X_eval_list.append(val.item())
            else:
                X_eval_list.append(current_params[pid])
        X_eval = torch.tensor([[X_eval_list]], dtype=torch.float32) # Shape (1, 1, 4)

        # Generate the single modified image using the internal UINT8 original
        # generate_image expects uint8 [0, 255], returns float tensor likely [0, 255]
        modified_img_internal_float = generate_image(X_eval, img_torch_internal_uint8).squeeze() # (C, H, W) float [0, 255]

        try:
            # Compute metrics and score using the internal float modified image
            # Prepare batch: float [0, 255] -> add batch dim -> (1, C, H, W) float [0, 255]
            img_batch_float_0_255 = modified_img_internal_float.unsqueeze(0)

            sh_raw, de_raw, cl_raw, to_raw, co_raw = loss_func._compute_all_metrics(img_batch_float_0_255) # Pass float [0, 255]
            sh, de, cl, to, co = loss_func._normalize_metrics(sh_raw, de_raw, cl_raw, to_raw, co_raw)

            other_metrics = torch.stack([de, cl, to, co], dim=1)
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


# Update fixed parameter input labels, ranges, and default values
@app.callback(
    Output('fixed-param-1-label', 'children'),
    Output('fixed-param-1-input', 'min'),
    Output('fixed-param-1-input', 'max'),
    Output('fixed-param-1-input', 'step'),
    Output('fixed-param-1-input', 'value'),
    Output('fixed-param-2-label', 'children'),
    Output('fixed-param-2-input', 'min'),
    Output('fixed-param-2-input', 'max'),
    Output('fixed-param-2-input', 'step'),
    Output('fixed-param-2-input', 'value'),
    Output('fixed-param-ids-store', 'data'), # Store the IDs being fixed
    Input('heatmap-x-axis', 'value'),
    Input('heatmap-y-axis', 'value'),
    State('slider-brightness', 'value'), # Get current slider values for defaults
    State('slider-contrast', 'value'),
    State('slider-saturation', 'value'),
    State('slider-hue', 'value'),
)
def update_fixed_param_inputs(x_axis_id, y_axis_id, cur_br, cur_co, cur_sa, cur_hu):
    if not x_axis_id or not y_axis_id:
        return no_update # Should not happen with clearable=False

    current_param_values = {'brightness': cur_br, 'contrast': cur_co, 'saturation': cur_sa, 'hue': cur_hu}
    param_name_map = dict(zip(param_ids, param_names))

    fixed_ids = [p_id for p_id in param_ids if p_id not in [x_axis_id, y_axis_id]]

    if len(fixed_ids) != 2:
        # This might happen briefly if x_axis == y_axis, handle gracefully
        logging.warning(f"Error: Expected 2 fixed params, found {len(fixed_ids)} (X={x_axis_id}, Y={y_axis_id})")
        # Return default labels/values for first two params as fallback
        fixed_ids = [pid for pid in param_ids if pid not in [param_ids[0], param_ids[1]]]
        if len(fixed_ids) != 2: fixed_ids = param_ids[2:] # Absolute fallback


    p1_id = fixed_ids[0]
    p2_id = fixed_ids[1]

    p1_name = param_name_map[p1_id]
    p2_name = param_name_map[p2_id]
    p1_bounds = param_bounds_map[p1_id]
    p2_bounds = param_bounds_map[p2_id]
    p1_default = current_param_values[p1_id] # Use current slider value as default
    p2_default = current_param_values[p2_id] # Use current slider value as default

    step = 0.01 # Standard step

    fixed_param_ids_data = {'p1': p1_id, 'p2': p2_id}

    return (
        f"Fixed {p1_name}:", p1_bounds[0], p1_bounds[1], step, p1_default,
        f"Fixed {p2_name}:", p2_bounds[0], p2_bounds[1], step, p2_default,
        fixed_param_ids_data
    )


# Generate 2D heatmap plot
@app.callback(
    Output('loading-heatmap-plot', 'children'),
    Output('heatmap-calculation-status', 'children'),
    Input('plot-heatmap-button', 'n_clicks'),
    State('heatmap-x-axis', 'value'),
    State('heatmap-y-axis', 'value'),
    State('fixed-param-1-input', 'value'),
    State('fixed-param-2-input', 'value'),
    State('fixed-param-ids-store', 'data'), # Get the IDs of the fixed params
    State('original-image-store', 'data'),
    State('loss-config-store', 'data'),
    State('heatmap-resolution-slider', 'value'), # <-- ADD THIS STATE
    prevent_initial_call=True,
)
def generate_heatmap_plot(
    n_clicks, x_id, y_id, fixed_val_1, fixed_val_2, fixed_ids_data,
    stored_image_data, loss_config,
    heatmap_resolution # <-- ADD THIS ARGUMENT
):
    start_time = time.time()
    status = ""
    placeholder_fig = go.Figure().update_layout(
        title="Select Axes, Set Fixed Values, and Click 'Generate Heatmap'",
         xaxis_title="X Parameter", yaxis_title="Y Parameter"
    )
    param_name_map = dict(zip(param_ids, param_names))

    # --- Validation (add heatmap_resolution check) ---
    if not all([n_clicks, x_id, y_id, fixed_val_1 is not None, fixed_val_2 is not None,
                fixed_ids_data, stored_image_data, loss_config,
                heatmap_resolution is not None]): # <-- CHECK RESOLUTION
        logging.warning("Heatmap plot called without necessary inputs.")
        return [dcc.Graph(figure=placeholder_fig)], "Missing inputs."

    # --- Ensure resolution is an integer ---
    try:
        heatmap_resolution = int(heatmap_resolution)
        if heatmap_resolution < 2: # Enforce minimum practical size
             heatmap_resolution = 2
    except (ValueError, TypeError):
        logging.error(f"Invalid heatmap resolution value received: {heatmap_resolution}")
        error_fig = go.Figure().update_layout(title="Invalid heatmap resolution value.")
        return [dcc.Graph(figure=error_fig)], "Error: Invalid resolution."

    if x_id == y_id:
         error_fig = go.Figure().update_layout(title="X and Y axes cannot be the same parameter.")
         return [dcc.Graph(figure=error_fig)], "Error: X=Y axis."

    fixed_id_1 = fixed_ids_data.get('p1')
    fixed_id_2 = fixed_ids_data.get('p2')
    if not fixed_id_1 or not fixed_id_2:
        error_fig = go.Figure().update_layout(title="Error identifying fixed parameters.")
        return [dcc.Graph(figure=error_fig)], "Error: Fixed param IDs missing."


    # --- 1. Get Base Image and Config ---
    try:
        img_str = stored_image_data['image_b64']
        img_pil = Image.open(io.BytesIO(base64.b64decode(img_str))).convert('RGB')
        img_np_hwc = np.array(img_pil)
        img_torch_orig_uint8 = torch.from_numpy(img_np_hwc).permute(2, 0, 1).byte()
    except Exception as e:
        logging.exception("Error decoding stored image for heatmap:")
        error_fig = go.Figure().update_layout(title=f"Error loading image: {e}")
        return [dcc.Graph(figure=error_fig)], f"Error loading image: {e}"

    internal_size = loss_config.get('size', DEFAULT_IMG_SIZE)
    img_torch_internal_uint8 = resize(img_torch_orig_uint8, [internal_size, internal_size], antialias=True)
    logging.info(f"Heatmap using internal image: {img_torch_internal_uint8.shape}")

    loss_func = ImageAestheticsLoss(
        original_image=img_torch_internal_uint8,
        k_levels=loss_config.get('k', DEFAULT_PYRAMID_LEVELS),
        tone_u=loss_config.get('u', DEFAULT_TONE_U_PARAM),
        tone_o=loss_config.get('o', DEFAULT_TONE_O_PARAM),
        negate=True # We want the score
    )

    # --- 2. Define Parameter Grid ---
    x_bounds = param_bounds_map[x_id]
    y_bounds = param_bounds_map[y_id]
    x_range = torch.linspace(x_bounds[0], x_bounds[1], steps=heatmap_resolution)
    y_range = torch.linspace(y_bounds[0], y_bounds[1], steps=heatmap_resolution)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij') # Output shape (GRID_SIZE, GRID_SIZE)

    # Flatten grids for batch processing
    flat_x = grid_x.flatten() # Shape (N*M,)
    flat_y = grid_y.flatten() # Shape (N*M,)
    num_points = flat_x.shape[0]

    # --- 3. Construct Parameter Batch ---
    # Create the full 4D parameter batch tensor (N*M, 1, 4)
    # The order matters and must match the param_ids list
    param_batch = torch.zeros((num_points, 1, 4), dtype=torch.float32)
    fixed_params = {fixed_id_1: float(fixed_val_1), fixed_id_2: float(fixed_val_2)}

    for i, p_id in enumerate(param_ids):
        if p_id == x_id:
            param_batch[:, 0, i] = flat_x
        elif p_id == y_id:
            param_batch[:, 0, i] = flat_y
        elif p_id == fixed_id_1:
            param_batch[:, 0, i] = fixed_params[fixed_id_1]
        elif p_id == fixed_id_2:
            param_batch[:, 0, i] = fixed_params[fixed_id_2]
        else:
             # This case should not happen if logic is correct
             logging.error(f"Unexpected parameter ID '{p_id}' encountered during batch construction.")
             error_fig = go.Figure().update_layout(title="Internal error building parameter batch.")
             return [dcc.Graph(figure=error_fig)], "Internal Error."

    logging.info(f"Generated parameter batch of shape: {param_batch.shape}")

    # --- 4. Calculate Scores (Vectorized) ---
    calc_start = time.time()
    try:
        # Generate all modified images in a batch
        # Input: (N*M, 1, 4), uint8 (C,H,W) -> Output: (N*M, 1, C, H, W) float [0, 255]
        modified_images_batch_nq = generate_image(param_batch, img_torch_internal_uint8)
        logging.info(f"Generated modified image batch shape (N,q,C,H,W): {modified_images_batch_nq.shape}")

        # --- FIX: Squeeze the q=1 dimension ---
        # _compute_all_metrics expects (B, C, H, W) where B = N*M
        modified_images_batch = modified_images_batch_nq.squeeze(1)
        logging.info(f"Shape passed to _compute_all_metrics (B,C,H,W): {modified_images_batch.shape}")
        # --------------------------------------

        # Compute metrics and scores for the entire batch
        # Input: (N*M, C, H, W) float [0, 255]
        sh_raw, de_raw, cl_raw, to_raw, co_raw = loss_func._compute_all_metrics(modified_images_batch) # Outputs are (N*M,)
        sh, de, cl, to, co = loss_func._normalize_metrics(sh_raw, de_raw, cl_raw, to_raw, co_raw)

        other_metrics = torch.stack([de, cl, to, co], dim=1) # (N*M, 4)
        mean_others = other_metrics.mean(dim=1) # (N*M,)
        scores_flat = sh * mean_others # (N*M,)
        scores_flat = torch.nan_to_num(scores_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape scores back into a 2D grid
        scores_grid = scores_flat.reshape(heatmap_resolution, heatmap_resolution).cpu().numpy() # (GRID_SIZE, GRID_SIZE)

    except Exception as e:
        logging.exception("Error during batch score calculation for heatmap:")
        error_fig = go.Figure().update_layout(title=f"Error calculating scores: {e}")
        return [dcc.Graph(figure=error_fig)], f"Calculation Error: {e}"

    calc_time = time.time() - calc_start
    logging.info(f"Heatmap score calculation took {calc_time:.2f}s")
    status = f"Calculation time: {calc_time:.2f}s"

    # --- 5. Create Heatmap Plot ---
    fig = go.Figure(data=go.Heatmap(
        z=scores_grid.T,  # Transpose: z[row, col] should correspond to y[row] and x[col]; our scores_grid[i, j] corresponds to x_range[i] and y_range[j]; transposing makes z.T[j, i] correspond to y_range[j] and x_range[i]
        x=x_range.cpu().numpy(),
        y=y_range.cpu().numpy(),
        colorscale='Viridis', # Or choose another colorscale
        colorbar_title='Aesthetic Score'
    ))

    x_name = param_name_map[x_id]
    y_name = param_name_map[y_id]
    fixed_name_1 = param_name_map[fixed_id_1]
    fixed_name_2 = param_name_map[fixed_id_2]

    fig.update_layout(
        title=f"Score Heatmap: {y_name} vs. {x_name}<br>(Fixed: {fixed_name_1}={fixed_val_1:.2f}, {fixed_name_2}={fixed_val_2:.2f})",
        xaxis_title=f"{x_name}",
        yaxis_title=f"{y_name}",
        yaxis_autorange='reversed', # Often intuitive for heatmaps
        margin=dict(l=60, r=20, t=80, b=50),
    )

    total_time = time.time() - start_time
    logging.info(f"Total heatmap generation time: {total_time:.2f}s")

    return [dcc.Graph(figure=fig)], status


# Callback to update PCP total points info text
@app.callback(
    Output('pcp-total-points-info', 'children'),
    Input('pcp-points-slider', 'value')
)
def update_pcp_total_points_info(points_per_dim):
    if points_per_dim:
        total_points = int(points_per_dim) ** 4
        return f"Total points to calculate: {points_per_dim}⁴ = {total_points}"
    return ""


# Callback to generate Parallel Coordinates Plot
@app.callback(
    Output('loading-pcp-plot', 'children'),
    Output('pcp-calculation-status', 'children'),
    Input('generate-pcp-button', 'n_clicks'),
    State('pcp-points-slider', 'value'),
    State('original-image-store', 'data'),
    State('loss-config-store', 'data'),
    prevent_initial_call=True,
)
def generate_pcp_plot(n_clicks, points_per_dim, stored_image_data, loss_config):
    start_time = time.time()
    status = ""
    placeholder_fig = go.Figure().update_layout(
        title="Adjust Settings and Click 'Generate Plot'",
        # Provide dummy axes to avoid empty look
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    param_name_map = dict(zip(param_ids, param_names))

    if not all([n_clicks, points_per_dim, stored_image_data, loss_config]):
        logging.warning("PCP plot called without necessary inputs.")
        return [dcc.Graph(figure=placeholder_fig)], "Missing inputs."

    try:
        n_points = int(points_per_dim)
        if n_points < 2:
            n_points = 2
        total_points = n_points ** 4
    except (ValueError, TypeError):
         logging.error(f"Invalid points per dimension value: {points_per_dim}")
         error_fig = go.Figure().update_layout(title="Invalid points per dimension.")
         return [dcc.Graph(figure=error_fig)], "Error: Invalid input."

    # --- 1. Get Base Image and Config ---
    try:
        img_str = stored_image_data['image_b64']
        img_pil = Image.open(io.BytesIO(base64.b64decode(img_str))).convert('RGB')
        img_np_hwc = np.array(img_pil)
        img_torch_orig_uint8 = torch.from_numpy(img_np_hwc).permute(2, 0, 1).byte()
    except Exception as e:
        logging.exception("Error decoding stored image for PCP:")
        error_fig = go.Figure().update_layout(title=f"Error loading image: {e}")
        return [dcc.Graph(figure=error_fig)], f"Error loading image: {e}"

    internal_size = loss_config.get('size', DEFAULT_IMG_SIZE)
    img_torch_internal_uint8 = resize(img_torch_orig_uint8, [internal_size, internal_size], antialias=True)
    logging.info(f"PCP using internal image: {img_torch_internal_uint8.shape}")

    try:
        loss_func = ImageAestheticsLoss(
            original_image=img_torch_internal_uint8,
            k_levels=loss_config.get('k', DEFAULT_PYRAMID_LEVELS),
            tone_u=loss_config.get('u', DEFAULT_TONE_U_PARAM),
            tone_o=loss_config.get('o', DEFAULT_TONE_O_PARAM),
            negate=True # We want the score
        )
    except Exception as e:
        logging.exception("Error initializing loss function for PCP:")
        error_fig = go.Figure().update_layout(title=f"Error initializing loss function: {e}")
        return [dcc.Graph(figure=error_fig)], f"Loss function error: {e}"

    # --- 2. Define Parameter Grid (4D) ---
    grid_creation_start = time.time()
    ranges = []
    for p_id in param_ids:
        bounds = param_bounds_map[p_id]
        ranges.append(torch.linspace(bounds[0], bounds[1], steps=n_points))

    # Create the 4D grid - meshgrid output order needs careful handling for flattening
    # grid_b, grid_c, grid_s, grid_h = torch.meshgrid(ranges[0], ranges[1], ranges[2], ranges[3], indexing='ij')
    # Instead of meshgrid which can be memory intensive for >2D, use itertools.product
    import itertools
    param_combinations = list(itertools.product(*ranges)) # List of tuples [(b0,c0,s0,h0), (b0,c0,s0,h1), ...]
    param_combinations_torch = torch.tensor(param_combinations, dtype=torch.float32) # Shape (total_points, 4)
    logging.info(f"Created parameter combinations tensor: {param_combinations_torch.shape}")

    # Reshape for generate_image: (total_points, 1, 4)
    param_batch = param_combinations_torch.unsqueeze(1)
    logging.info(f"Grid creation time: {time.time() - grid_creation_start:.2f}s")

    # --- 3. Calculate Scores (Vectorized) ---
    calc_start = time.time()
    all_scores = []
    batch_size = 256 # Process in smaller batches to avoid OOM errors if total_points is large
    try:
        for i in range(0, total_points, batch_size):
            batch_indices = slice(i, min(i + batch_size, total_points))
            current_param_batch = param_batch[batch_indices]

            # Generate modified images for the current batch
            modified_images_batch_nq = generate_image(current_param_batch, img_torch_internal_uint8)
            modified_images_batch = modified_images_batch_nq.squeeze(1) # (batch, C, H, W) float [0, 255]

            # Compute metrics and scores for the current batch
            sh_raw, de_raw, cl_raw, to_raw, co_raw = loss_func._compute_all_metrics(modified_images_batch) # (batch,)
            sh, de, cl, to, co = loss_func._normalize_metrics(sh_raw, de_raw, cl_raw, to_raw, co_raw)
            other_metrics = torch.stack([de, cl, to, co], dim=1) # (batch, 4)
            mean_others = other_metrics.mean(dim=1) # (batch,)
            scores = sh * mean_others # (batch,)
            scores = torch.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0) # Use -1 for invalid scores for PCP vis
            all_scores.append(scores.cpu())
            logging.info(f"Processed batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")

        scores_flat = torch.cat(all_scores) # Shape (total_points,)

    except Exception as e:
        logging.exception(f"Error during batch score calculation for PCP (batch starting {i}):")
        error_fig = go.Figure().update_layout(title=f"Error calculating scores: {e}")
        return [dcc.Graph(figure=error_fig)], f"Calculation Error: {e}"

    calc_time = time.time() - calc_start
    logging.info(f"PCP score calculation took {calc_time:.2f}s for {total_points} points.")
    status = f"Calculated {total_points} points in {calc_time:.2f}s."

    # --- 4. Create Parallel Coordinates Plot ---
    plot_start = time.time()

    # Prepare dimensions for Parcoords
    dimensions = []
    # Add parameter dimensions
    for i, p_id in enumerate(param_ids):
        dimensions.append(dict(
            range=param_bounds_map[p_id],
            label=param_name_map[p_id],
            values=param_combinations_torch[:, i].tolist() # Pass flattened values for this dim
        ))
    # Add score dimension
    score_min, score_max = scores_flat.min().item(), scores_flat.max().item()
    # Add a small buffer to range if min/max are close, handle NaN case where min/max might be -1
    score_range_buffer = (score_max - score_min) * 0.05 if score_max > score_min else 0.1
    score_range = [score_min - score_range_buffer, score_max + score_range_buffer]

    dimensions.append(dict(
        range=score_range,
        label='Aesthetic Score',
        values=scores_flat.tolist(),
    ))

    # Create the figure
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=scores_flat.tolist(), # Color lines by score
            colorscale='viridis',      # Choose a colorscale
            showscale=True,
            cmin=score_min,
            cmax=score_max,
            colorbar=dict(title='Score')
        ),
        dimensions=dimensions
    ))

    fig.update_layout(
        title=f"Parallel Coordinates Plot ({total_points} points)",
        margin=dict(l=60, r=60, t=80, b=50), # Adjust margins for labels
    )
    logging.info(f"PCP figure creation time: {time.time() - plot_start:.2f}s")

    total_time = time.time() - start_time
    logging.info(f"Total PCP generation time: {total_time:.2f}s")

    return [dcc.Graph(figure=fig)], status


# --- Run the App ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Dash app running on http://127.0.0.1:8050/")
    # Set debug=False for production or if experiencing reload issues
    # Set debug=True for development (enables hot-reloading and error pages)
    app.run(debug=True, port=8050)