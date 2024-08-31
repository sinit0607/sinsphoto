from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from rembg import remove
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import io
import base64
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

def detect_main_object(image: Image.Image) -> tuple:
    """Detect the main object in the image and return its bounding box."""
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        return (x, y, w, h)
    else:
        return (0, 0, image.size[0], image.size[1])

def apply_3d_effect(image: Image.Image) -> Image.Image:
    """Apply a 3D effect to the image by creating a shadow."""
    image_np = np.array(image)
    shadow_offset = 10

    # Create an empty array for the shadow
    shadow = np.zeros_like(image_np)

    # Apply shadow offset to create a 3D effect look
    shadow[shadow_offset:, shadow_offset:] = image_np[:-shadow_offset, :-shadow_offset]

    shadow_image = Image.fromarray(shadow)

    # Applying GaussianBlur to soften the shadow effect
    shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(radius=5))

    # Combine original image with shadow to create a 3D effect
    final_image = Image.alpha_composite(shadow_image.convert('RGBA'), image.convert('RGBA'))

    return final_image

def image_to_base64(image: Image.Image) -> str:
    """Convert an image to a base64-encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    """Remove the background from the uploaded image and return a base64-encoded result."""
    try:
        # Load the image
        image = Image.open(file.file).convert('RGBA')

        # Log the size of the uploaded file
        logging.info(f"Processing file {file.filename}")

        # Remove the background without adding any shadow
        removed_background_image = remove(image)

        # Convert the image to a base64 string
        image_base64 = image_to_base64(removed_background_image)

        return JSONResponse({"message": "Success", "image_data": image_base64})
    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {e}")
        return {"error": str(e)}

@app.post("/apply-3d-effect/")
async def create_3d_effect(file: UploadFile = File(...)):
    """Apply a 3D effect to the uploaded image and return a base64-encoded result."""
    try:
        # Load the image with transparent background
        image = Image.open(file.file).convert('RGBA')

        # Apply the 3D effect
        image_with_3d = apply_3d_effect(image)

        # Convert the image to a base64 string
        image_base64 = image_to_base64(image_with_3d)

        return JSONResponse({"message": "Success", "image_data": image_base64})
    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {e}")
        return {"error": str(e)}

@app.post("/replace-background/")
async def replace_background(main_image_file: UploadFile = File(...), background_image_file: UploadFile = File(...)):
    """Replace the background of the main image with a new background and return a base64-encoded result."""
    try:
        # Load the main image and the new background image
        main_image = Image.open(main_image_file.file).convert('RGBA')
        background_image = Image.open(background_image_file.file).convert('RGBA')

        # Remove the background of the main image
        removed_background_image = remove(main_image)

        # Resize the new background image to match the main image size
        background_image = background_image.resize(removed_background_image.size)

        # Composite the main image onto the new background
        final_image = Image.alpha_composite(background_image, removed_background_image)

        # Convert the image to a base64 string
        image_base64 = image_to_base64(final_image)

        return JSONResponse({"message": "Success", "image_data": image_base64})
    except Exception as e:
        logging.error(f"Error processing files {main_image_file.filename} and {background_image_file.filename}: {e}")
        return {"error": str(e)}

@app.post("/enhance-photo/")
async def enhance_photo(file: UploadFile = File(...), brightness: float = 1.2, contrast: float = 1.2,
                        sharpness: float = 1.5):
    """Enhance the uploaded image by adjusting brightness, contrast, and sharpness, then return a base64-encoded result."""
    try:
        # Load the image
        image = Image.open(file.file).convert('RGBA')

        # Enhance the image
        enhancer_brightness = ImageEnhance.Brightness(image)
        enhanced_image = enhancer_brightness.enhance(brightness)

        enhancer_contrast = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer_contrast.enhance(contrast)

        enhancer_sharpness = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer_sharpness.enhance(sharpness)

        # Convert the image to a base64 string
        image_base64 = image_to_base64(enhanced_image)

        return JSONResponse({"message": "Success", "image_data": image_base64})
    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {e}")
        return {"error": str(e)}

