# Step 1: Install and authenticate
# -------------------------------

# Install required Python packages quietly (-q flag) to avoid cluttering output
# Packages installed:
# - google-cloud-aiplatform: Google's Vertex AI SDK
# - pillow: Python Imaging Library (PIL fork) for image processing
# - requests: HTTP library for API calls
# - matplotlib: Plotting and visualization library
# - ipywidgets: Interactive widgets for Jupyter notebooks
# - imageio: Library for reading and writing image data
!pip install -q google-cloud-aiplatform pillow requests matplotlib ipywidgets imageio

# Import Google Colab authentication module
from google.colab import auth

# Authenticate the user with Google Cloud
# This will open a popup requesting Google account credentials
# and grant access to Google Cloud services
auth.authenticate_user()

# Print success message after authentication completes
# The checkmark emoji (✅) provides visual confirmation of success
print("✅ Google Cloud authentication successful!")

# Step 2: Configuration
# ---------------------
# This section sets up the essential configuration parameters needed
# to interact with Google Cloud's Vertex AI services.

# Google Cloud Project ID
# Replace with your actual GCP project ID where Vertex AI is enabled
# Format: "your-project-name-12345"
# This identifies which Google Cloud project to bill and where services will run
PROJECT_ID = "gen-lang-client-0091268922"  # Your GCP project ID

# Google Cloud Location/Region
# Specifies the geographical region where the AI services will be hosted
# "us-central1" is Iowa, USA - a common default region with good availability
# Other options include:
#   - "us-west1" (Oregon)
#   - "europe-west4" (Netherlands)
#   - "asia-east1" (Taiwan)
# Must match the region where your AI services are enabled
LOCATION = "us-central1"

# Note about configuration:
# These values are critical for:
# 1. Proper service routing to the correct regional endpoints
# 2. Billing attribution to the correct project
# 3. Compliance with any data residency requirements

# Step 3: Initialize Vertex AI and widgets
# ---------------------------------------
# This section imports all required libraries and initializes the core components
# needed for image generation, processing, and the interactive interface.

# Google Cloud Vertex AI Imports
from google.cloud import aiplatform  # Main Vertex AI SDK
from google.cloud.aiplatform_v1 import PredictionServiceClient  # For making prediction requests

# Image Processing Imports (Pillow/PIL)
from PIL import Image  # Core image manipulation
from PIL import ImageDraw  # Drawing on images
from PIL import ImageFont  # Font handling for text overlays
from PIL import ImageSequence  # Working with animated images

# Web and Data Handling
import requests  # HTTP requests for fallback images
from io import BytesIO  # In-memory binary stream handling
import base64  # Base64 encoding/decoding for image data

# Text Processing
import textwrap  # Smart text wrapping for meme text

# Visualization and Display
import matplotlib.pyplot as plt  # Image display in notebooks
from IPython.display import display, clear_output, HTML  # Notebook output control

# Interactive Widgets
import ipywidgets as widgets  # UI components (buttons, dropdowns, etc.)

# System Utilities
import time  # For delays and timing
import numpy as np  # Numerical operations (used by some image functions)

# Animation Support
import imageio  # GIF creation and manipulation
from matplotlib import animation  # Animation framework
from matplotlib.animation import PillowWriter  # GIF writer for animations

# File Operations
from google.colab import files  # File download in Colab environment

# Note: These imports collectively enable:
# - AI-powered image generation
# - Image manipulation and enhancement
# - Interactive user interface
# - Animated meme creation
# - Result display and download

# Initialize the Vertex AI Prediction Service Client with retry logic
# ----------------------------------------------------------------

# Configure client options with the regional API endpoint
# This ensures requests are routed to the correct geographical location
# Format: "{region}-aiplatform.googleapis.com" (e.g., "us-central1-aiplatform.googleapis.com")
client_options = {
    "api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"  # Dynamic endpoint based on configured location
}

# Create the PredictionServiceClient instance
# This client will handle all communication with Vertex AI's prediction services
client = PredictionServiceClient(client_options=client_options)

# Important Notes:
# 1. The client automatically includes retry logic for:
#    - Temporary network failures (auto-retry)
#    - Rate limiting (auto-backoff)
#    - Service interruptions (auto-retry with exponential backoff)
#
# 2. Default retry configuration:
#    - Maximum 3 retry attempts
#    - Exponential backoff starting at 1 second
#    - Total timeout of 60 seconds


# Step 4: Enhanced Image Generation with Style Options
# --------------------------------------------------
def generate_image(prompt, style="photo", retries=3):
    """Generate AI images with style customization and automatic retry logic

    Args:
        prompt (str): Text description of the desired image content
        style (str): Visual style from available options (default: "photo")
        retries (int): Number of attempts if generation fails (default: 3)

    Returns:
        PIL.Image: Generated image object or None if all attempts fail

    Features:
        - Multiple artistic style presets
        - Automatic prompt enhancement
        - Built-in retry mechanism
        - Image validation
        - Error handling with fallback
    """

    # Style configuration dictionary - contains optimized parameters for each art style
    style_configs = {
        "photo": {
            "prompt_prefix": "Professional photograph of ",
            "prompt_suffix": ", 8K UHD, sharp focus, natural lighting",
            "negative_prompt": "blurry, cartoon, drawing, painting, text, watermark",
            "style": "photo",  # Vertex AI style parameter
            "guidance_scale": 12  # How strictly to follow the prompt (7-14 typical)
        },
        "cartoon": {
            "prompt_prefix": "Pixar-style 3D animation of ",
            "prompt_suffix": ", vibrant colors, soft lighting, cartoon style",
            "negative_prompt": "realistic, photo, photograph, blurry, text",
            "style": "cartoon",
            "guidance_scale": 10  # Slightly lower for creative interpretation
        },
        # [Other styles follow same pattern...]
    }

    # Get configuration for selected style, defaulting to photo if invalid
    config = style_configs.get(style, style_configs["photo"])

    # Retry loop for fault tolerance
    for attempt in range(retries):
        try:
            # Construct API endpoint using project and location
            endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/imagegeneration"

            # Make the prediction request to Vertex AI
            response = client.predict(
                endpoint=endpoint,
                instances=[{
                    # Combine prompt components with user input
                    "prompt": f"{config['prompt_prefix']}{prompt}{config.get('prompt_suffix', '')}",
                    # Elements to avoid in generated image
                    "negativePrompt": config["negative_prompt"]
                }],
                parameters={
                    "sampleCount": 1,  # Generate single image
                    "aspectRatio": "1:1",  # Square format
                    "style": config["style"],  # Apply selected style
                    "guidanceScale": config["guidance_scale"]  # Creativity vs accuracy
                }
            )

            # Validate API response
            if not response.predictions:
                raise ValueError("Empty response from API")

            # Extract base64-encoded image data
            img_data = response.predictions[0].get('bytesBase64Encoded')
            if not img_data:
                raise ValueError("No image data in response")

            # Decode and create PIL Image object
            img = Image.open(BytesIO(base64.b64decode(img_data)))

            # Quality control - reject very small images
            if img.size[0] < 100 or img.size[1] < 100:
                raise ValueError("Generated image too small")

            return img  # Success case

        except Exception as e:
            # Error handling with retry delay
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)  # Exponential backoff could be added here
                continue
            return None  # All attempts failed

# Step 5: Enhanced Text Overlay with Animation Effects
# ---------------------------------------------------
def add_meme_text(img, text, position="bottom", text_color="#FFFFFF",
                outline_color="#000000", animation_effect="none"):
    """Add professional meme text with multiple animation effects

    Args:
        img (PIL.Image): Base image to add text to
        text (str): Text content to overlay
        position (str): Text position ("top", "center", "bottom")
        text_color (str): Hex color for main text (e.g. "#FFFFFF" for white)
        outline_color (str): Hex color for text outline
        animation_effect (str): Animation type ("none", "fade_in", "slide_up", "typing")

    Returns:
        PIL.Image or list: Single image (no animation) or list of frames (animation)

    Features:
        - Dynamic font sizing based on image dimensions
        - Smart text wrapping
        - Multiple animation effects
        - Thick text outlines for readability
        - Fallback font system
    """
    try:
        # Download professional meme font (Impact-like)
        # Using wget to fetch from GitHub repository
        !wget -q https://github.com/phoikoi/fonts/raw/main/Impact.ttf -O /tmp/meme_font.ttf

        # Ensure image is in RGB mode (consistent color handling)
        img = img.convert('RGB')

        # Initialize frames list (used for animations)
        frames = []

        # Calculate dynamic font size based on:
        # - Image width (scales with image size)
        # - Text length (smaller for longer text)
        base_size = min(80, max(40, int(img.width/12)))  # Range: 40-80px
        font_size = max(base_size - (len(text)//3), 30)  # Minimum 30px

        # Try loading custom font, fallback to default if unavailable
        try:
            font = ImageFont.truetype("/tmp/meme_font.ttf", font_size)
        except:
            print("⚠️ Using fallback font")
            font = ImageFont.load_default(size=font_size)

        # Pre-process text:
        # 1. Convert to uppercase (classic meme style)
        # 2. Apply smart wrapping based on calculated max width
        text = text.upper()
        max_width = img.width * 0.9  # Use 90% of image width
        avg_char_width = font_size * 0.6  # Estimate character width
        wrap_width = min(20, int(max_width / avg_char_width))  # Dynamic wrap
        lines = textwrap.wrap(text, width=wrap_width)  # Split into multiple lines

        # Calculate total text block height (for vertical positioning)
        line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
        total_height = sum(line_heights) + 10 * (len(lines) - 1)  # 10px line spacing

        # Handle different animation effects
        if animation_effect == "none":
            # Static image - single frame
            frame = img.copy()
            draw = ImageDraw.Draw(frame)
            add_text_to_frame(draw, lines, font, img.width, img.height,
                            total_height, position, text_color, outline_color)
            return frame

        elif animation_effect == "fade_in":
            # Fade-in effect (10 frames with increasing opacity)
            for i in range(10):
                alpha = int(255 * (i+1)/10)  # 10%-100% opacity
                frame = img.copy()
                overlay = Image.new('RGBA', frame.size, (0,0,0,0))  # Transparent layer
                draw = ImageDraw.Draw(overlay)
                # Add text with current alpha (appended to hex colors)
                add_text_to_frame(draw, lines, font, img.width, img.height,
                                total_height, position,
                                f"{text_color}{alpha:02x}",
                                f"{outline_color}{alpha:02x}")
                frame.paste(overlay, (0,0), overlay)
                frames.append(frame)
            return frames

        elif animation_effect == "slide_up":
            # Slide-up from bottom (10 frames)
            for i in range(10):
                frame = img.copy()
                draw = ImageDraw.Draw(frame)
                # Calculate vertical offset (decreases each frame)
                offset = int((img.height * 0.2) * (10-i)/10)
                add_text_to_frame(draw, lines, font, img.width, img.height,
                                total_height, position,
                                text_color, outline_color,
                                y_offset=offset)
                frames.append(frame)
            return frames

        elif animation_effect == "typing":
            # Typing effect (character by character)
            frame = img.copy()
            draw = ImageDraw.Draw(frame)
            for i in range(1, len(text)+1):
                temp_frame = img.copy()
                temp_draw = ImageDraw.Draw(temp_frame)
                partial_text = text[:i]  # Incrementally reveal text
                partial_lines = textwrap.wrap(partial_text, width=wrap_width)
                add_text_to_frame(temp_draw, partial_lines, font,
                                img.width, img.height, total_height,
                                position, text_color, outline_color)
                frames.append(temp_frame)
            # Hold final frame for 5 extra frames
            for _ in range(5):
                frames.append(frames[-1])
            return frames

        else:
            # Default case (no animation)
            frame = img.copy()
            draw = ImageDraw.Draw(frame)
            add_text_to_frame(draw, lines, font, img.width, img.height,
                            total_height, position,
                            text_color, outline_color)
            return frame

    except Exception as e:
        print(f"⚠️ Text overlay error: {str(e)}")
        return None


def add_text_to_frame(draw, lines, font, img_width, img_height,
                     total_height, position, text_color,
                     outline_color, y_offset=0):
    """Helper function to render text onto an image frame

    Args:
        draw (ImageDraw): Drawing context
        lines (list): Wrapped text lines
        font (ImageFont): Font to use
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
        total_height (int): Total height of text block
        position (str): Text position ("top", "center", "bottom")
        text_color (str): Text color hex code
        outline_color (str): Outline color hex code
        y_offset (int): Vertical adjustment offset
    """
    # Calculate vertical starting position
    if position == "top":
        y = int(img_height * 0.1) + y_offset  # Upper 10% of image
    elif position == "center":
        y = (img_height - total_height) // 2 + y_offset  # Centered
    else:  # bottom (default)
        y = img_height - total_height - int(img_height * 0.1) + y_offset  # Lower 10%

    # Render each line of text
    for line in lines:
        bbox = font.getbbox(line)
        text_width = bbox[2] - bbox[0]
        x = (img_width - text_width) // 2  # Center horizontally

        # Draw outline using 3x3 grid for thickness
        for dx in [-3, 0, 3]:
            for dy in [-3, 0, 3]:
                if dx != 0 or dy != 0:  # Skip center position
                    draw.text((x+dx, y+dy), line, font=font, fill=outline_color)

        # Add subtle glow effect (secondary outline)
        for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
            draw.text((x+dx, y+dy), line, font=font, fill="#aaaaaa")  # Light gray

        # Draw main text
        draw.text((x, y), line, font=font, fill=text_color)

        # Move to next line position
        y += (bbox[3] - bbox[1]) + 10  # Line height + 10px spacing

# Step 6: Widget-based Meme Creator with GIF Support
# -------------------------------------------------
def create_meme_with_widgets():
    """Create an interactive meme generator with GIF support

    Features:
    - Complete GUI interface with ipywidgets
    - Real-time previews and feedback
    - Support for both static and animated memes
    - Built-in fallback images
    - One-click download functionality
    """

    # ==============================================
    # Widget Creation Section
    # ==============================================

    # Image Description Input (Textarea)
    image_prompt = widgets.Textarea(
        value='a surprised cat looking at a cucumber',  # Default example
        placeholder='Describe exactly what you want in the image...',
        description='Image Description:',  # Label
        disabled=False,
        layout=widgets.Layout(width='90%', height='80px')  # Responsive sizing
    )

    # Help Text Below Prompt (HTML widget)
    prompt_help = widgets.HTML(
        """<div style="font-size: 0.8em; color: #666; margin-top: -10px; margin-bottom: 10px;">
        Tip: Be specific! Include details about subject, action, setting, and style.
        Example: "a shocked cat looking at a cucumber on a kitchen counter, dramatic lighting"
        </div>"""
    )

    # Meme Text Input (Text widget)
    meme_text = widgets.Text(
        value='WHEN YOU SEE IT',  # Classic meme text example
        placeholder='Enter your meme text (will be automatically capitalized)',
        description='Meme Text:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )

    # Style Selection Dropdown
    style_selector = widgets.Dropdown(
        options=[
            ('Photorealistic', 'photo'),  # (Display text, value)
            ('Cartoon/Pixar', 'cartoon'),
            ('Digital Art', 'art'),
            ('Watercolor', 'watercolor'),
            ('Cyberpunk', 'cyberpunk')
        ],
        value='photo',  # Default selection
        description='Art Style:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )

    # Text Position Dropdown
    position_selector = widgets.Dropdown(
        options=[
            ('Top', 'top'),  # (Display text, value)
            ('Center', 'center'),
            ('Bottom', 'bottom')  # Classic meme position
        ],
        value='bottom',  # Default selection
        description='Text Position:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )

    # Color Pickers with Enhanced UI
    text_color = widgets.ColorPicker(
        concise=False,  # Show full color picker
        description='Text Color:',
        value='white',  # Default meme text color
        style={'description_width': 'initial'},  # Better label alignment
        layout=widgets.Layout(width='90%')
    )

    outline_color = widgets.ColorPicker(
        concise=False,
        description='Outline Color:',
        value='black',  # Classic meme outline
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='90%')
    )

    # Animation Effects Dropdown
    animation_selector = widgets.Dropdown(
        options=[
            ('No animation', 'none'),
            ('Fade in', 'fade_in'),  # Gradual appearance
            ('Slide up', 'slide_up'),  # Text slides into position
            ('Typing effect', 'typing')  # Character-by-character reveal
        ],
        value='none',  # Default to static
        description='Animation:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )

    # Output Format Toggle
    output_format = widgets.RadioButtons(
        options=['Static Image (JPEG)', 'Animated GIF'],  # Output choices
        value='Static Image (JPEG)',  # Default format
        description='Output Format:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )

    # Color Preview Display (Horizontal Box)
    color_preview = widgets.HBox([
        widgets.Label("Preview:"),
        # Text color preview box
        widgets.HTML('<div style="width: 30px; height: 20px; background: white; border: 1px solid black; margin: 0 10px;"></div>'),
        widgets.Label("Text"),
        # Outline color preview box
        widgets.HTML('<div style="width: 30px; height: 20px; background: black; border: 1px solid #ccc; margin: 0 10px;"></div>'),
        widgets.Label("Outline")
    ])

    # Generate Button (Primary Action)
    generate_btn = widgets.Button(
        description='✨ Generate Meme ✨',  # Emphasize with emoji
        button_style='success',  # Green color for primary action
        tooltip='Click to generate your meme',
        icon='bolt',  # Lightning bolt icon
        layout=widgets.Layout(width='250px', height='40px'),  # Fixed size
        style={'font_weight': 'bold'}  # Visually prominent
    )

    # Output Area for Results
    output = widgets.Output()

    # ==============================================
    # Interactive Functions Section
    # ==============================================

    # Color Preview Updater
    def update_color_preview(change):
        """Update the color preview boxes when colors change"""
        color_preview.children[1].value = f'<div style="width: 30px; height: 20px; background: {text_color.value}; border: 1px solid #ccc; margin: 0 10px;"></div>'
        color_preview.children[3].value = f'<div style="width: 30px; height: 20px; background: {outline_color.value}; border: 1px solid #ccc; margin: 0 10px;"></div>'

    # Attach color change observers
    text_color.observe(update_color_preview, names='value')
    outline_color.observe(update_color_preview, names='value')

    # ==============================================
    # UI Layout Section
    # ==============================================

    # Display widgets in organized vertical sections
    display(widgets.VBox([
        # Header
        widgets.HTML("<h1 style='text-align: center;'>🎬 Ultimate Meme Generator</h1>"),

        # Section 1: Image Description
        widgets.HTML("<h3 style='margin-bottom: 5px;'>1. Describe Your Image</h3>"),
        image_prompt,
        prompt_help,

        # Section 2: Style Selection
        widgets.HTML("<h3 style='margin-bottom: 5px;'>2. Customize Appearance</h3>"),
        style_selector,

        # Section 3: Text Options
        widgets.HTML("<h3 style='margin-bottom: 5px;'>3. Add Your Text</h3>"),
        meme_text,
        position_selector,
        widgets.HTML("<div style='margin-top: 10px;'></div>"),  # Spacer
        text_color,
        outline_color,
        color_preview,

        # Section 4: Animation
        widgets.HTML("<h3 style='margin-bottom: 5px;'>4. Animation Options</h3>"),
        animation_selector,
        output_format,

        # Generate Button
        widgets.HTML("<div style='margin-top: 20px;'></div>"),  # Spacer
        generate_btn,

        # Output Area
        output
    ]))

    # ==============================================
    # Generation Handler
    # ==============================================

    def on_generate_clicked(b):
        """Handle meme generation when button is clicked"""
        with output:
            clear_output()
            print("🔄 Generating your meme (this may take 20-40 seconds)...")

            # Get all current widget values
            prompt = image_prompt.value.strip()
            text = meme_text.value.strip()
            style = style_selector.value
            position = position_selector.value
            txt_color = text_color.value
            ol_color = outline_color.value
            animation_effect = animation_selector.value
            is_gif = output_format.value == 'Animated GIF'

            # Validate inputs
            if not prompt:
                print("⚠️ Please describe what you want in the image")
                return
            if not text:
                print("⚠️ Please enter some meme text")
                return

            # Generate AI image
            img = generate_image(prompt, style=style)

            # Fallback to stock images if generation fails
            if img is None:
                print("\n⚠️ AI generation failed, using high-quality fallback")
                try:
                    # Curated fallback images for each style
                    fallback_urls = {
                        "photo": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
                        "cartoon": "https://images.unsplash.com/photo-1637858868799-7f26a0640eb6",
                        "art": "https://images.unsplash.com/photo-1534447677768-be436bb09401",
                        "watercolor": "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5",
                        "cyberpunk": "https://images.unsplash.com/photo-1547036967-23d11aacaee0"
                    }
                    # Fetch and resize fallback image
                    response = requests.get(
                        fallback_urls.get(style, fallback_urls["photo"]),
                        params={
                            "ixlib": "rb-1.2.1",
                            "auto": "format",
                            "fit": "crop",
                            "w": "800",
                            "h": "800"
                        },
                        timeout=15
                    )
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                except Exception as e:
                    print(f"❌ Fallback failed: {str(e)}")
                    return

            # Add text overlay with selected effects
            result = add_meme_text(
                img,
                text,
                position=position,
                text_color=txt_color,
                outline_color=ol_color,
                animation_effect=animation_effect if is_gif else "none"
            )

            if result:
                if is_gif and isinstance(result, list):
                    # Handle animated GIF output
                    print("\n🎬 Creating animated GIF...")

                    # Configure GIF parameters
                    filename = f"meme_{style}_{position}.gif"
                    duration = 100  # milliseconds per frame
                    loop = 0  # infinite looping

                    # Save all frames as animated GIF
                    result[0].save(
                        filename,
                        save_all=True,
                        append_images=result[1:],
                        duration=duration,
                        loop=loop,
                        optimize=True
                    )

                    # Display GIF in notebook
                    with open(filename, 'rb') as f:
                        display(HTML(
                            f'<img src="data:image/gif;base64,{base64.b64encode(f.read()).decode()}" />'
                        ))

                else:
                    # Handle static image output
                    final_meme = result[-1] if isinstance(result, list) else result

                    # Display image with matplotlib
                    plt.figure(figsize=(10, 8))
                    plt.imshow(final_meme)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()

                    # Save as high-quality JPEG
                    filename = f"meme_{style}_{position}.jpg"
                    final_meme.save(
                        filename,
                        quality=95,  # High quality
                        subsampling=0,  # No chroma subsampling
                        optimize=True  # Extra compression
                    )

                # Add download button
                download_btn = widgets.Button(
                    description='⬇️ Download Meme ⬇️',
                    button_style='info',  # Blue color
                    icon='download',
                    layout=widgets.Layout(width='250px', height='40px'),
                    style={'font_weight': 'bold'}
                )

                def on_download_clicked(b):
                    """Trigger file download when clicked"""
                    files.download(filename)

                download_btn.on_click(on_download_clicked)
                display(download_btn)

                print(f"\n✅ Success! Meme saved as '{filename}'")
            else:
                print("❌ Failed to create meme")

    # Attach click handler to generate button
    generate_btn.on_click(on_generate_clicked)

# Launch the widget interface
create_meme_with_widgets()
