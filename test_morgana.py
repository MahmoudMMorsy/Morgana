import os
import time
import torch
import numpy as np
from PIL import Image
from morgana_model import MorganaPixelGenerator
from morgana_pipeline import MorganaPipeline, PALETTES

def test_morgana_model_forward():
    """
    Test that the base PyTorch generative model instantiates
    and performs a forward pass correctly, yielding the expected output dimensions.
    """
    model = MorganaPixelGenerator(embedding_dim=384, base_channels=64)
    model.eval()  # Put model in eval mode to allow batch size of 1
    dummy_input = torch.randn(1, 384)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 3, 128, 128), f"Expected shape (1, 3, 128, 128), got {output.shape}"
    assert isinstance(output, torch.Tensor)

def test_morgana_pipeline_initialization():
    """
    Test that the generation pipeline loads cleanly.
    """
    pipeline = MorganaPipeline()
    assert pipeline.pipe is not None

def test_morgana_pipeline_generation():
    """
    Test generating a full pixel-art image from a prompt.
    """
    pipeline = MorganaPipeline()

    start_time = time.time()
    img = pipeline.generate(
        prompt="retro gameboy hero with a glowing sword",
        palette_name="gameboy",
        target_size=(256, 256),
        dither_strength=1.0,
        dither_type="bayer"
    )
    elapsed = time.time() - start_time

    assert isinstance(img, Image.Image), "Output should be a PIL Image"
    assert img.size == (256, 256), f"Expected size (256, 256), got {img.size}"
    print(f"\n[INFO] Generated pixel art in {elapsed:.3f} seconds.")

def test_palette_and_quantization():
    """
    Test that the image post-processor successfully maps pixels
    to the restricted palette colors exactly.
    """
    pipeline = MorganaPipeline()

    # Create a simple test gradient array
    raw_img = np.zeros((128, 128, 3), dtype=np.uint8)
    for y in range(128):
        raw_img[y, :, 0] = int((y / 128.0) * 255.0)  # Red gradient

    processed_img = pipeline.process_pixel_art(
        img_np=raw_img,
        palette_name="gameboy",
        target_size=(160, 144),
        dither_strength=0.0  # Turn off dithering for exact matching
    )

    processed_np = np.array(processed_img)
    unique_colors = np.unique(processed_np.reshape(-1, 3), axis=0)

    expected_palette = PALETTES["gameboy"]

    for color in unique_colors:
        color_tuple = tuple(color)
        assert color_tuple in expected_palette, f"Pixel color {color_tuple} not found in GameBoy palette: {expected_palette}"
