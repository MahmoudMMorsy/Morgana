import os
import io
import base64
from flask import Flask, request, render_template_string, jsonify
from morgana_pipeline import MorganaPipeline, PALETTES

app = Flask(__name__)

# Initialize Morgana Pipelines for both modes
# We keep local_pipeline initialized instantly; genuine_pipeline can be initialized on demand to save memory!
local_pipeline = MorganaPipeline(use_sd=False)
genuine_pipeline = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MorganaPixel - Tiny & Mighty Pixel Art Generator</title>
    <!-- Press Start 2P Font for Retro Vibes -->
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --retro-dark: #0f0f1c;
            --retro-card: #1c1c34;
            --retro-accent: #ff007f;
            --retro-cyan: #00ffff;
            --retro-text: #ffffff;
            --retro-muted: #8c8cab;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--retro-dark);
            color: var(--retro-text);
            font-family: 'Tajawal', sans-serif;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        .retro-font {
            font-family: 'Press Start 2P', monospace;
            line-height: 1.6;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            width: 100%;
            max-width: 900px;
        }

        header h1 {
            color: var(--retro-accent);
            font-size: 1.8rem;
            text-shadow: 3px 3px 0px var(--retro-cyan);
            margin-bottom: 10px;
        }

        header p {
            color: var(--retro-muted);
            font-size: 1.1rem;
        }

        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            width: 100%;
            max-width: 900px;
            justify-content: center;
        }

        .card {
            background-color: var(--retro-card);
            border: 4px solid var(--retro-text);
            box-shadow: 6px 6px 0px var(--retro-accent);
            padding: 25px;
            border-radius: 8px;
            flex: 1 1 400px;
            max-width: 500px;
        }

        .preview-card {
            flex: 1 1 300px;
            max-width: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            text-align: center;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: var(--retro-cyan);
        }

        input[type="text"], select, input[type="number"], range {
            width: 100%;
            padding: 12px;
            background-color: var(--retro-dark);
            border: 2px solid var(--retro-muted);
            color: var(--retro-text);
            font-size: 1rem;
            border-radius: 4px;
            outline: none;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus, select:focus, input[type="number"]:focus {
            border-color: var(--retro-accent);
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .slider-container input {
            flex: 1;
        }

        .slider-value {
            min-width: 40px;
            font-weight: bold;
            color: var(--retro-accent);
        }

        button {
            width: 100%;
            background-color: var(--retro-accent);
            color: white;
            border: 3px solid var(--retro-text);
            padding: 15px;
            font-weight: bold;
            font-size: 1.1rem;
            cursor: pointer;
            box-shadow: 4px 4px 0px var(--retro-cyan);
            transition: all 0.1s ease;
            text-transform: uppercase;
        }

        button:active {
            transform: translate(2px, 2px);
            box-shadow: 2px 2px 0px var(--retro-cyan);
        }

        #image-preview {
            max-width: 100%;
            border: 4px solid var(--retro-text);
            box-shadow: 5px 5px 0px var(--retro-cyan);
            image-rendering: pixelated;
            display: none;
            margin-top: 15px;
        }

        .placeholder-box {
            border: 3px dashed var(--retro-muted);
            padding: 40px 20px;
            width: 100%;
            color: var(--retro-muted);
            font-family: 'Press Start 2P', monospace;
            font-size: 0.8rem;
        }

        .loader {
            border: 6px solid var(--retro-card);
            border-top: 6px solid var(--retro-accent);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            display: none;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .footer {
            margin-top: 40px;
            text-align: center;
            color: var(--retro-muted);
            font-size: 0.9rem;
            width: 100%;
            max-width: 900px;
            border-top: 2px dashed var(--retro-muted);
            padding-top: 20px;
        }
    </style>
</head>
<body>

    <header>
        <h1 class="retro-font">MORGANA PIXEL</h1>
        <p>نموذج خارق خفيف الوزن لتوليد صور البيكسل آرت بالذكاء الاصطناعي على أي جهاز وبدون كارت شاشة</p>
    </header>

    <div class="container">
        <!-- Settings Form Card -->
        <div class="card">
            <form id="gen-form">
                <div class="form-group">
                    <label for="prompt">الوصف باللغة العربية أو الإنجليزية (Prompt)</label>
                    <input type="text" id="prompt" name="prompt" required placeholder="مثال: castle at sunset" value="castle at sunset with a beautiful glowing sun">
                </div>

                <div class="form-group">
                    <label for="mode">نمط محرك التوليد (Generation Engine)</label>
                    <select id="mode" name="mode">
                        <option value="local" selected>Ultra-fast Offline Neural (محلي فائق السرعة - 0 ميجا تحميل)</option>
                        <option value="genuine">Genuine Stable Diffusion (نمط التوليد العصبي الحقيقي < 500MB)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="palette">نمط لوحة الألوان (Retro Palette)</label>
                    <select id="palette" name="palette">
                        <option value="gameboy" selected>GameBoy Classic (4 colors of green)</option>
                        <option value="famicom">Famicom/NES (Classic 8-bit)</option>
                        <option value="gba">GameBoy Advance (Vibrant 16-bit)</option>
                        <option value="pico8">Pico-8 Fantasy Console</option>
                        <option value="c64">Commodore 64 Classic</option>
                        <option value="cyberpunk">Cyberpunk Neon</option>
                        <option value="retro_sunset">Outrun Retro Sunset</option>
                        <option value="monochrome">Monochrome Black & White</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="dither_type">نوع التظليل (Dithering Style)</label>
                    <select id="dither_type" name="dither_type">
                        <option value="bayer" selected>Bayer Ordered (Classic hardware matrix)</option>
                        <option value="floyd">Floyd-Steinberg (Organic error diffusion)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="dither">قوة التظليل (Dithering Strength)</label>
                    <div class="slider-container">
                        <input type="range" id="dither" name="dither" min="0" max="2" step="0.1" value="1.0">
                        <span id="dither-val" class="slider-value">1.0</span>
                    </div>
                </div>

                <div class="form-group">
                    <label for="size">الدقة الناتجة (Output Resolution)</label>
                    <select id="size" name="size">
                        <option value="256">256 x 256 (Fastest)</option>
                        <option value="512" selected>512 x 512 (Recommended)</option>
                        <option value="768">768 x 768</option>
                        <option value="1024">1024 x 1024</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="seed">كود البذرة العشوائية (Seed - اختياري)</label>
                    <input type="number" id="seed" name="seed" placeholder="ترك فارغاً للتوليد العشوائي">
                </div>

                <button type="submit" class="retro-font">توليد الصورة / Generate</button>
            </form>
        </div>

        <!-- Preview Card -->
        <div class="card preview-card">
            <div id="placeholder" class="placeholder-box">
                [ المعاينة تظهر هنا ]<br><br>[ READY TO GENERATE ]
            </div>
            <div id="loader" class="loader"></div>
            <img id="image-preview" src="" alt="Morgana Generated Masterpiece">
        </div>
    </div>

    <div class="footer">
        <p class="retro-font" style="font-size: 0.75rem; margin-bottom: 10px;">MORGANA PIXEL MODEL v1.0.0</p>
        <p>مصمم ومطور بلغة بايثون للتشغيل الفوري باستهلاك لا يذكر من الذاكرة والـ CPU.</p>
    </div>

    <script>
        const ditherSlider = document.getElementById('dither');
        const ditherVal = document.getElementById('dither-val');
        ditherSlider.addEventListener('input', (e) => {
            ditherVal.textContent = e.target.value;
        });

        const form = document.getElementById('gen-form');
        const placeholder = document.getElementById('placeholder');
        const loader = document.getElementById('loader');
        const preview = document.getElementById('image-preview');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            placeholder.style.display = 'none';
            preview.style.display = 'none';
            loader.style.display = 'block';

            const formData = new FormData(form);
            const data = {
                prompt: formData.get('prompt'),
                mode: formData.get('mode'),
                palette: formData.get('palette'),
                dither_type: formData.get('dither_type'),
                dither: parseFloat(formData.get('dither')),
                size: parseInt(formData.get('size')),
                seed: formData.get('seed') ? parseInt(formData.get('seed')) : null
            };

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    preview.src = 'data:image/png;base64,' + result.image;
                    preview.style.display = 'block';
                } else {
                    alert('Error generating image: ' + result.error);
                    placeholder.style.display = 'block';
                }
            } catch (err) {
                alert('Connection error: ' + err);
                placeholder.style.display = 'block';
            } finally {
                loader.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    global genuine_pipeline
    try:
        data = request.json or {}
        prompt = data.get('prompt', 'castle at sunset')
        mode = data.get('mode', 'local')
        palette = data.get('palette', 'gameboy')
        dither = data.get('dither', 1.0)
        dither_type = data.get('dither_type', 'bayer')
        size_val = data.get('size', 512)
        seed = data.get('seed')

        # Select pipeline based on mode
        if mode == 'genuine':
            # Initialize genuine pipeline on demand to keep initial startup instantaneous!
            if genuine_pipeline is None:
                genuine_pipeline = MorganaPipeline(use_sd=True)
            pipe = genuine_pipeline
        else:
            pipe = local_pipeline

        img = pipe.generate(
            prompt=prompt,
            palette_name=palette,
            target_size=(size_val, size_val),
            dither_strength=dither,
            dither_type=dither_type,
            seed=seed
        )

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_str
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
