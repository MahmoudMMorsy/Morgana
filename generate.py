#!/usr/bin/env python3
import argparse
import sys
import os
import time
from morgana_pipeline import MorganaPipeline, PALETTES

def print_banner():
    banner = r"""
========================================================================
   __  ___                                  ___ _          _
  /  |/  |___  _ _ __ _ __ _ _ _  __ _     / _ \ |___ _  _| |
 /_ /_/__/ _ \| '_/ _` / _` | ' \/ _` |   |  _// \_ / | || |_|
(_)(_)(_)\___/|_| \__, \__,_|_||_\__,_|   |_|  |_/|___\_,_(_)
                  |___/|___/
========================================================================
       MorganaPixel - Tiny & Mighty Retro Pixel Art Generator
      مورجانا بيكسل - نموذج توليد صور بيكسل آرت خارق بأقل إمكانيات
========================================================================
    """
    print(banner)

def main():
    print_banner()

    palette_options = list(PALETTES.keys())

    parser = argparse.ArgumentParser(
        description="MorganaPixel: Generate stunning GameBoy, Famicom, and GBA pixel art completely on CPU with zero graphics card requirements!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("prompt", type=str, help="The text prompt to generate (Arabic or English!) / الوصف المطلوب لتوليد الصورة")
    parser.add_argument("-p", "--palette", type=str, default="gameboy", choices=palette_options,
                        help=f"Retro hardware palette choice. Available: {palette_options}")
    parser.add_argument("-o", "--output", type=str, default="output.png", help="Path to save the generated image")
    parser.add_argument("-s", "--size", type=int, default=512, help="Output image size (width & height in pixels)")
    parser.add_argument("-d", "--dither", type=float, default=1.0, help="Dithering intensity strength (0.0 to 2.0)")
    parser.add_argument("-t", "--dither_type", type=str, default="bayer", choices=["bayer", "floyd"],
                        help="Dithering style: 'bayer' (classic hardware grid) or 'floyd' (organic error diffusion)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--use_sd", action="store_true",
                        help="Use real pre-trained 'segmind/tiny-sd' + 'TAESD' neural network model (< 500MB) for infinite variety")

    args = parser.parse_args()

    print(f"[*] Prompt: '{args.prompt}'")
    print(f"[*] Selected Palette: {args.palette.upper()}")
    print(f"[*] Dithering style: {args.dither_type.upper()} (Strength: {args.dither})")
    print(f"[*] Resolution: {args.size}x{args.size}")
    if args.seed is not None:
        print(f"[*] Seed: {args.seed}")
    if args.use_sd:
        print("[*] Mode: Genuine Deep-Learning Stable Diffusion Model / نمط التوليد العصبي الحقيقي")
    else:
        print("[*] Mode: Ultra-lightweight Offline Neural Mapper / نمط التوليد المحلي فائق السرعة")

    start_time = time.time()

    try:
        pipeline = MorganaPipeline(use_sd=args.use_sd)
    except Exception as e:
        print(f"[-] Error initializing MorganaPipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print("[*] Generating retro masterpiece / جاري توليد التحفة الفنية...")
    try:
        image = pipeline.generate(
            prompt=args.prompt,
            palette_name=args.palette,
            target_size=(args.size, args.size),
            dither_strength=args.dither,
            dither_type=args.dither_type,
            seed=args.seed
        )

        image.save(args.output)
        elapsed_time = time.time() - start_time

        print("\n========================================================================")
        print(f"[+] SUCCESS! Saved generated pixel art to: {args.output}")
        print(f"[+] Total generation time: {elapsed_time:.3f} seconds!")
        print(f"[+] File Size on disk: {os.path.getsize(args.output) / 1024:.2f} KB")
        print("========================================================================")
        print("  مبروك! تم توليد الصورة بنجاح وبسرعة خارقة وبدون استهلاك كارت الشاشة نهائياً.")
        print("========================================================================")

    except Exception as e:
        print(f"[-] Error during generation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
