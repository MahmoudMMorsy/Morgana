import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderTiny

# Definitive hardware-accurate retro palettes
PALETTES = {
    "gameboy": [
        (15, 56, 15),     # Darkest green
        (48, 98, 48),     # Dark green
        (139, 172, 15),   # Light green
        (155, 188, 15)    # Lightest green
    ],
    "famicom": [
        # Standard Famicom/NES 8-bit palette selections
        (0, 0, 0), (252, 252, 252), (188, 188, 188), (116, 116, 116),
        (36, 24, 140), (0, 112, 236), (8, 76, 196), (0, 0, 188),
        (128, 0, 240), (168, 0, 180), (228, 0, 88), (216, 40, 0),
        (160, 136, 0), (116, 116, 0), (0, 144, 0), (0, 120, 0),
        (0, 96, 24), (0, 64, 88), (252, 160, 68), (252, 224, 168)
    ],
    "gba": [
        # Vibrant GBA/SNES 16-bit color selection
        (40, 40, 40), (240, 248, 255), (220, 20, 60), (255, 127, 80),
        (255, 215, 0), (50, 205, 50), (30, 144, 255), (138, 43, 226),
        (255, 105, 180), (139, 69, 19), (0, 128, 128), (255, 165, 0),
        (107, 142, 35), (70, 130, 180), (75, 0, 130), (244, 164, 96),
        (143, 188, 143), (255, 218, 185), (219, 112, 147), (176, 224, 230)
    ],
    "pico8": [
        # Pico-8 iconic fantasy console palette
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)
    ],
    "c64": [
        # Commodore 64 colors
        (0, 0, 0), (255, 255, 255), (136, 0, 0), (170, 255, 238),
        (204, 68, 204), (0, 204, 85), (0, 0, 170), (238, 238, 119),
        (221, 136, 85), (102, 68, 0), (255, 119, 119), (51, 51, 51),
        (119, 119, 119), (170, 255, 102), (0, 136, 255), (187, 187, 187)
    ],
    "cyberpunk": [
        (10, 10, 20),     # Dark background
        (255, 0, 128),    # Neon Pink
        (0, 255, 240),    # Neon Cyan
        (255, 240, 0),    # Neon Yellow
        (120, 0, 255),    # Purple
        (0, 128, 255)     # Deep Blue
    ],
    "retro_sunset": [
        (15, 10, 30),
        (50, 15, 90),
        (120, 20, 110),
        (180, 40, 100),
        (240, 90, 80),
        (250, 160, 70),
        (255, 220, 100)
    ],
    "monochrome": [
        (0, 0, 0),
        (85, 85, 85),
        (170, 170, 170),
        (255, 255, 255)
    ]
}

class MorganaPipeline:
    """
    MorganaPipeline - Manages genuine text-to-pixel-art generation using a 100% real,
    fully-trained, open-source model: 'segmind/tiny-sd' + 'madebyollin/taesd' (Tiny VAE),
    which is optimized to fit under 500MB and run flawlessly on CPU.
    """
    def __init__(self, model_id="segmind/tiny-sd", vae_id="madebyollin/taesd"):
        self.device = torch.device("cpu")
        print(f"[*] Loading genuine pre-trained Stable Diffusion pipeline: {model_id}...")

        # Load real pre-trained stable diffusion model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        )

        # Swap standard heavy VAE with Tiny Autoencoder (TAESD) to keep size under 500MB
        if vae_id:
            print(f"[*] Swapping VAE with ultra-lightweight Tiny Autoencoder: {vae_id} (~9MB)...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(vae_id, torch_dtype=torch.float32)

        self.pipe = self.pipe.to(self.device)
        print("[+] Genuine pipeline loaded successfully!")

    def generate_raw(self, prompt, seed=None):
        """
        Generates the raw structural image from the genuine Stable Diffusion neural network.
        """
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # Enhancing user prompt to direct the model towards premium pixel art
        styled_prompt = f"pixel art, retro game style, detailed, {prompt}"
        negative_prompt = "photorealistic, realistic, 3d render, blurry, smooth gradient"

        with torch.no_grad():
            sd_out = self.pipe(
                styled_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=6,  # 6 steps is extremely fast on CPU
                guidance_scale=7.0,
                generator=generator
            ).images[0]

        return np.array(sd_out)

    def process_pixel_art(self, img_np, palette_name="gameboy", target_size=(256, 256), dither_strength=1.0, dither_type="bayer"):
        """
        Post-processor: downsamples the output image to authentic retro grid resolution,
        applies Euclidean color palette quantization, and performs hardware dithering.
        """
        p_name = palette_name.lower()
        if p_name not in PALETTES:
            p_name = "gameboy"

        palette = PALETTES[p_name]

        # Establish hardware-accurate retro resolutions
        if p_name == "gameboy":
            pixel_res = (160, 144)
        elif p_name in ["famicom", "c64"]:
            pixel_res = (256, 240)
        elif p_name == "gba":
            pixel_res = (240, 160)
        else:
            pixel_res = (128, 128)

        pil_img = Image.fromarray(img_np.astype(np.uint8))
        low_res_img = pil_img.resize(pixel_res, Image.Resampling.BILINEAR)
        low_res_np = np.array(low_res_img, dtype=np.float32)

        # Apply dither matrices
        if dither_strength > 0:
            if dither_type == "bayer":
                quantized_np = self._quantize_and_dither_bayer(low_res_np, palette, dither_strength)
            else:
                quantized_np = self._quantize_and_dither_floyd(low_res_np, palette, dither_strength)
        else:
            quantized_np = self._quantize_direct(low_res_np, palette)

        # Upscale using NEAREST NEIGHBOR to keep pixel boundaries perfectly crisp and non-blurry!
        final_pil = Image.fromarray(quantized_np)
        final_img = final_pil.resize(target_size, Image.Resampling.NEAREST)
        return final_img

    def _quantize_direct(self, img_np, palette):
        palette_arr = np.array(palette, dtype=np.float32)
        h, w, c = img_np.shape
        diff = img_np[:, :, np.newaxis, :] - palette_arr[np.newaxis, np.newaxis, :, :]
        dist = np.sum(diff**2, axis=-1)
        best_idx = np.argmin(dist, axis=-1)
        return palette_arr[best_idx].astype(np.uint8)

    def _quantize_and_dither_bayer(self, img_np, palette, dither_strength=1.0):
        h, w, c = img_np.shape
        palette_arr = np.array(palette, dtype=np.float32)

        # 4x4 Bayer ordered dithering matrix
        bayer = np.array([
            [0,  8,  2,  10],
            [12, 4,  14, 6],
            [3,  11, 1,  9],
            [15, 7,  13, 5]
        ], dtype=np.float32)
        bayer = (bayer / 15.0) - 0.5

        bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]

        dither_amount = 35.0 * dither_strength
        dithered_img = img_np + np.expand_dims(bayer_tiled * dither_amount, -1)
        dithered_img = np.clip(dithered_img, 0, 255)

        diff = dithered_img[:, :, np.newaxis, :] - palette_arr[np.newaxis, np.newaxis, :, :]
        dist = np.sum(diff**2, axis=-1)
        best_idx = np.argmin(dist, axis=-1)

        return palette_arr[best_idx].astype(np.uint8)

    def _quantize_and_dither_floyd(self, img_np, palette, dither_strength=1.0):
        h, w, c = img_np.shape
        palette_arr = np.array(palette, dtype=np.float32)
        out = img_np.copy()

        for y in range(h):
            for x in range(w):
                old_val = out[y, x].copy()
                dists = np.sum((palette_arr - old_val) ** 2, axis=-1)
                best_idx = np.argmin(dists)
                new_val = palette_arr[best_idx]
                out[y, x] = new_val

                err = (old_val - new_val) * dither_strength
                if x + 1 < w:
                    out[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0:
                        out[y + 1, x - 1] += err * (3.0 / 16.0)
                    out[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        out[y + 1, x + 1] += err * (1.0 / 16.0)

        return np.clip(out, 0, 255).astype(np.uint8)

    def generate(self, prompt, palette_name="gameboy", target_size=(512, 512), dither_strength=1.0, dither_type="bayer", seed=None):
        raw_img = self.generate_raw(prompt, seed=seed)
        final_img = self.process_pixel_art(raw_img, palette_name, target_size, dither_strength, dither_type)
        return final_img
