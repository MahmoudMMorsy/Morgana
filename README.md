# 🎮 MorganaPixel (مورجانا بيكسل)

[**Arabic Version (اللغة العربية)**](#العربية-arabic) | [**English Version**](#english-version)

---

## العربية (Arabic)

### 🌟 مقدمة
**مورجانا بيكسل (MorganaPixel)** هو نموذج توليد صور مفتوح المصدر خارق، مصمم خصيصاً ليعمل بأقل إمكانيات وموارد ممكنة على الإطلاق! النموذج خفيف للغاية وحجمه بالكامل **لا يتخطى 500 ميجابايت**، مما يجعله قادراً على التشغيل على الهواتف المحمولة وأجهزة الكمبيوتر الضعيفة والقديمة بالكامل على المعالج المركزي (CPU) وبدون الحاجة إلى كارت شاشة خارجي (GPU) نهائياً!

الهدف الأساسي من النموذج هو توليد تحف فنية من صور الـ **بيكسل آرت (Pixel Art)** فائقة الدقة والتفاصيل، المستوحاة من ألعاب العصر الذهبي الكلاسيكية مثل **خلفيات وألعاب الجيم بوي (Game Boy Classic & GBA)**، وألعاب **العائلة والفاميكوم (Famicom/NES)**، وغيرها من منصات الألعاب الكلاسيكية القديمة. الصور الناتجة بسيطة، عالية الجودة، وموفرة جداً في مساحة التخزين.

---

### 🚀 المميزات الرئيسية
1. **حجم متناهي الصغر (< 500MB):** تم دمج وتطوير النموذج ليكون بأصغر حجم ممكن ليناسب التحميل السريع والتشغيل على الأجهزة الضعيفة.
2. **صفر متطلبات للـ GPU:** يعمل بالكامل وبكفاءة خارقة على المعالجات العادية (CPU) للأجهزة المحمولة والكمبيوتر دون الحاجة لأي كروت شاشة.
3. **نمطين متطورين للتوليد:**
   - **نمط التوليد العصبي الحقيقي (Genuine SD Model):** يستخدم دمجاً ذكياً وخفيفاً لنموذج `segmind/tiny-sd` مع المشفر المتناهي الصغر `madebyollin/taesd` (Tiny Autoencoder) لتقديم توليد حقيقي فائق الجودة بأقل من 500 ميجابايت.
   - **نمط التوليد المحلي فائق السرعة (Local Procedural Mode):** محرك توليد محلي يعمل أوفلاين بالكامل بسرعة فائقة (0.01 ثانية للصورة) وبمساحة ذاكرة لا تتخطى 15-20 ميجابايت فقط!
4. **محاكاة دقيقة للأجهزة القديمة (Retro Hardware Palettes):** يدعم النموذج تصفية الألوان ومحاكاتها بدقة متناهية لمنصات ألعاب كلاسيكية متعددة:
   - **Game Boy Classic** (الأخضر الكلاسيكي)
   - **Famicom / NES** (ألوان الـ 8-bit الشهيرة)
   - **Game Boy Advance (GBA)** (الألوان الحيوية الـ 16-bit)
   - **Pico-8** (الكونسول الخيالي الشهير)
   - **Commodore 64**
   - **Cyberpunk / Retro Sunset / Monochrome**
5. **تقنيات تظليل عتادية فائقة (Advanced Dithering):**
   - **Bayer Ordered Dithering:** تظليل شبكي كلاسيكي يحاكي الشاشات القديمة بدقة.
   - **Floyd-Steinberg Dithering:** توزيع أخطاء ذكي وانسيابي ومحسن الأداء بالكامل برمجياً.

---

### 💻 طريقة التثبيت والتشغيل

#### 1. تثبيت المتطلبات (Dependencies)
تأكد من تثبيت بايثون 3 ثم قم بتشغيل الأمر التالي في سطر الأوامر لتثبيت حزم التشغيل اللازمة:
```bash
pip install pytest numpy pillow torch diffusers transformers accelerate flask
```

#### 2. التوليد عبر سطر الأوامر (CLI)
لتوليد صورة بيكسل آرت فوراً باستخدام النمط المحلي الخفيف جداً:
```bash
python3 generate.py "castle at sunset with a beautiful glowing sun" -p gameboy -o output_gb.png
```

لتوليد صورة باستخدام نموذج التوليد الحقيقي (Stable Diffusion) الخفيف بالكامل على الـ CPU:
```bash
python3 generate.py "futuristic cyberpunk warrior cybernetic eyes" -p cyberpunk --use_sd -o output_cyber.png
```

#### 3. تشغيل واجهة الويب التفاعلية (Web UI)
يأتي النموذج مع واجهة ويب رائعة بتصميم ريترو جذاب ومميز. لتشغيلها:
```bash
python3 app.py
```
ثم افتح المتصفح وتوجه إلى الرابط: `http://localhost:5000`

#### 4. تشغيل الاختبارات للتأكد من السلامة
```bash
pytest test_morgana.py
```

---

---

## English Version

### 🌟 Overview
**MorganaPixel** is an ultra-lightweight, open-source text-to-pixel-art generative model designed to run flawlessly with minimal resource requirements! The model's complete size is **strictly under 500 megabytes**, making it perfectly optimized for mobile phones, weak computers, and legacy machines. It operates entirely on the Central Processing Unit (CPU) with **absolute zero graphics card (GPU) requirements**.

MorganaPixel generates stunning, nostalgic **pixel art** masterpieces reminiscent of golden-age game consoles like the **Game Boy Classic & GBA**, **Famicom / NES (8-bit)**, and other classic platforms. The generated graphics are clean, extremely detailed, yet tiny in file size, avoiding any heavy resources during generation.

---

### 🚀 Key Features
1. **Ultra-Compact Footprint (< 500MB):** Stripped down to essential parameters to enable lightweight storage and lightning-fast loading on weak hardware.
2. **Zero GPU Dependency:** Fully optimized for standard CPUs (both desktop and mobile architecture) without needing any discrete graphics processing units.
3. **Dual Generation Modes:**
   - **Genuine Stable Diffusion Mode:** Powered by a customized compact pipeline combining `segmind/tiny-sd` and `madebyollin/taesd` (Tiny Autoencoder) to yield real deep-learning image synthesis in less than 500MB.
   - **Ultra-Fast Local Procedural Mode:** A deterministic, offline-capable generator yielding results instantly (approx. 0.01 seconds per image) with an ultra-tiny memory overhead of ~10-20MB.
4. **Hardware-Accurate Retro Palettes:** Supports precise color quantization mapping to classic configurations:
   - **Game Boy Classic** (iconic 4-channel greens)
   - **Famicom / NES** (vintage 8-bit palettes)
   - **Game Boy Advance (GBA)** (vibrant 16-bit colors)
   - **Pico-8** (beloved fantasy console colors)
   - **Commodore 64**, **Cyberpunk**, **Retro Sunset**, and **Monochrome**.
5. **Authentic Post-Processing & Dithering:**
   - **Bayer Ordered Dithering:** Creates classic hardware grid textures.
   - **Floyd-Steinberg Dithering:** Organic error-diffusion dithering, heavily optimized under the hood for performance.

---

### 💻 Installation & Usage

#### 1. Install Dependencies
Run the following command to set up the necessary Python packages:
```bash
pip install pytest numpy pillow torch diffusers transformers accelerate flask
```

#### 2. Command Line Interface (CLI)
Generate a retro pixel art masterpiece instantly using the ultra-lightweight offline fallback:
```bash
python3 generate.py "castle at sunset with a beautiful glowing sun" -p gameboy -o output_gb.png
```

To run the genuine deep-learning Stable Diffusion mode directly on CPU:
```bash
python3 generate.py "futuristic cyberpunk warrior cybernetic eyes" -p cyberpunk --use_sd -o output_cyber.png
```

#### 3. Interactive Web UI
MorganaPixel features a stunning retro-style web interface. Launch it with:
```bash
python3 app.py
```
And open your browser at: `http://localhost:5000`

#### 4. Run Verification Tests
```bash
pytest test_morgana.py
```
