import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_deck():
    prs = Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank layout
    
    # Palette
    C_BG = RGBColor(11, 15, 25)        # Dark navy slate
    C_CARD = RGBColor(22, 30, 49)      # Elevated surface
    C_CARD_BORDER = RGBColor(51, 65, 85)
    C_TEXT_MAIN = RGBColor(248, 250, 252) # Crisp white
    C_TEXT_MUTED = RGBColor(148, 163, 184) # Muted slate
    C_ACCENT_GREEN = RGBColor(16, 185, 129) # Emerald success
    C_ACCENT_RED = RGBColor(244, 63, 94)    # Crimson failure
    C_ACCENT_BLUE = RGBColor(56, 189, 248)  # Cyan highlight

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, category, title):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.name = "Arial"
        p0.font.color.rgb = C_ACCENT_BLUE
        
        p1 = tf.add_paragraph()
        p1.text = title
        p1.font.size = Pt(24)
        p1.font.bold = True
        p1.font.name = "Arial"
        p1.font.color.rgb = C_TEXT_MAIN

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1)
    
    # Badge
    b1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3), Inches(3.2), Inches(0.45))
    b1.fill.solid()
    b1.fill.fore_color.rgb = C_CARD
    b1.line.color.rgb = C_ACCENT_BLUE
    b1.line.width = Pt(1)
    p = b1.text_frame.paragraphs[0]
    p.text = "DATA-CENTRIC AI • OPTION B"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_ACCENT_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    # Title box
    tbox = s1.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.8))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "Robust Road Sign Recognition"
    p0.font.size = Pt(40)
    p0.font.bold = True
    p0.font.color.rgb = C_TEXT_MAIN
    
    p1 = tf.add_paragraph()
    p1.text = "Solving Optical & Environmental Blind Spots via Synthetic OpenCV Augmentation"
    p1.font.size = Pt(20)
    p1.font.color.rgb = C_ACCENT_GREEN
    
    p2 = tf.add_paragraph()
    p2.text = "ICT Project Defense • Duration: 3-5 Minutes • GitHub: AVELON1/ictteamprojeeect"
    p2.font.size = Pt(14)
    p2.font.color.rgb = C_TEXT_MUTED
    
    # Bottom key metrics card
    mc = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.5))
    mc.fill.solid()
    mc.fill.fore_color.rgb = C_CARD
    mc.line.color.rgb = C_CARD_BORDER
    mtf = mc.text_frame
    mtf.word_wrap = True
    mp = mtf.paragraphs[0]
    mp.text = "KEY TAKEAWAY: Models trained only on clean data drop to 72% accuracy and commit fatal false-positives under blur. Synthetic cv2 augmentation restores robust generalization (+12% to +20% boost) at zero additional inference latency."
    mp.font.size = Pt(13)
    mp.font.color.rgb = C_TEXT_MAIN
    
    s1.notes_slide.notes_text_frame.text = (
        "СЛАЙД 1 (Вступление, 30 сек):\n"
        "Здравствуйте, уважаемые члены комиссии! Наш проект посвящен Option B: «Устойчивое распознавание дорожных знаков в плохих условиях через синтетическое обогащение данных».\n"
        "В реальной жизни камеры беспилотников и ассистентов водителя сталкиваются с дождем, смазанным движением, сумерками и вибрацией. Обычные нейросети на чистых данных в таких условиях слепнут. Мы покажем, как с помощью OpenCV мы создали надежную модель и доказали эффективность Data-Centric подхода."
    )

    # ==========================================
    # SLIDE 2: Problem & Research Hypothesis
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2)
    add_header(s2, "Problem Formulation", "Why Standard Autonomous Vision Fails in Real Driving")
    
    # Left Card: The Vulnerability
    c_left = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = C_CARD
    c_left.line.color.rgb = C_ACCENT_RED
    c_left.line.width = Pt(1.5)
    ltf = c_left.text_frame
    ltf.word_wrap = True
    lp0 = ltf.paragraphs[0]
    lp0.text = "❌ The Fragility of Clean-Trained Models"
    lp0.font.bold = True
    lp0.font.size = Pt(16)
    lp0.font.color.rgb = C_ACCENT_RED
    
    items_l = [
        "Pristine Training Bias: Public datasets contain centered, sharp, daylight photos.",
        "Optical Defocus: Rain on lens, camera vibration, and high vehicle speed blur signs.",
        "Overconfident Hallucinations: When distorted, a clean model doesn't just guess — it becomes confidently wrong (e.g. 80.8% false confidence on Stop sign!).",
        "Safety Hazard: Misinterpreting 'Stop' as 'No Entry' or ignoring speed limits leads to critical autonomous vehicle collisions."
    ]
    for it in items_l:
        p = ltf.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(13)
        p.font.color.rgb = C_TEXT_MAIN
        
    # Right Card: The Solution (Option B)
    c_right = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = C_CARD
    c_right.line.color.rgb = C_ACCENT_GREEN
    c_right.line.width = Pt(1.5)
    rtf = c_right.text_frame
    rtf.word_wrap = True
    rp0 = rtf.paragraphs[0]
    rp0.text = "✅ The Option B Solution (Data-Centric AI)"
    rp0.font.bold = True
    rp0.font.size = Pt(16)
    rp0.font.color.rgb = C_ACCENT_GREEN
    
    items_r = [
        "Synthetic Hardening: Do NOT change the neural architecture or add heavy weights. Enrich the data distribution using OpenCV (cv2).",
        "Deterministic cv2 Transformations: Generate tilted (±15°), blurred (GaussianBlur), dusk/contrast altered variants.",
        "Domain Constraint: No horizontal flip (prevents ruining directional signs like 4.1.1 'Straight' or 3.20 'No Overtaking').",
        "Zero Extra Latency: The model remains lightweight (<10ms inference per frame on CPU)."
    ]
    for it in items_r:
        p = rtf.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(13)
        p.font.color.rgb = C_TEXT_MAIN
        
    s2.notes_slide.notes_text_frame.text = (
        "СЛАЙД 2 (Проблема и гипотеза, 45 сек):\n"
        "Почему вообще возник этот проект? Большинство нейросетей обучают на идеальных датасетах из интернета. Но на реальной дороге камера трясется, идет дождь, светят встречные фары.\n"
        "Наша гипотеза по Option B: нам не нужно усложнять саму нейросеть или использовать тяжелые трансформеры. Достаточно перед обучением синтетически ухудшить данные с помощью OpenCV — добавить размытие, наклоны и сумерки. Модель научится игнорировать помехи, сохранив мгновенную скорость работы."
    )

    # ==========================================
    # SLIDE 3: Development Journey & Dataset
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3)
    add_header(s3, "Pipeline & Methodology", "From Clean Dataset to Production Multi-Model Rig")
    
    # 3 Milestone Steps
    cards_data = [
        ("Step 1: Dataset Curating", "10 CIS Standard Classes", "Selected 10 vital road sign categories from database:\n• 1.22 Pedestrian, 2.1 Priority road\n• 2.4 Give way, 2.5 Stop\n• 3.1 No entry, 3.20 No overtaking\n• 3.27 No stopping, 4.1.1 Straight\n• 5.5 One-way, 6.4 Parking\nSplit: 3,326 clean train / 816 test.", Inches(0.8)),
        ("Step 2: Prototyping & Colab", "Validating CNN Baseline", "Initial experiments executed in Google Colab on GPU to verify loss convergence.\nTransitioned to dedicated local Python 3 pipeline inside isolated environment with TensorFlow 2.22 & OpenCV 5.0.\nEnsured 100% reproducibility.", Inches(4.8)),
        ("Step 3: Dual Training Rig", "Identical Weights & Arch", "Constructed two models with strictly identical ConvNet architectures:\n• Model 1 (Clean Baseline): Trained exclusively on 3,326 clean samples.\n• Model 2 (Option B Robust): Trained on 13,304 samples (clean + cv2 synthetic variants).", Inches(8.8))
    ]
    
    for title, subtitle, desc, left in cards_data:
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.8), Inches(3.7), Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD
        card.line.color.rgb = C_CARD_BORDER
        ctf = card.text_frame
        ctf.word_wrap = True
        p0 = ctf.paragraphs[0]
        p0.text = title
        p0.font.bold = True
        p0.font.size = Pt(15)
        p0.font.color.rgb = C_ACCENT_BLUE
        p1 = ctf.add_paragraph()
        p1.text = subtitle
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = C_TEXT_MUTED
        p2 = ctf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = C_TEXT_MAIN

    s3.notes_slide.notes_text_frame.text = (
        "СЛАЙД 3 (Пайплайн и процесс разработки, 45 сек):\n"
        "Как строилась наша работа с нуля:\n"
        "1. Мы подготовили датасет из 10 ключевых знаков СНГ (Пешеходный переход, Стоп, Главная дорога, Обгон запрещен и др.).\n"
        "2. Первые тесты провели в Google Colab на GPU, после чего перенесли весь пайплайн в чистый локальный проект на TensorFlow и OpenCV.\n"
        "3. Мы обучили две одинаковые модели: первую — только на чистых фото (3326 шт.), а вторую — на обогащенной выборке (13304 шт.)."
    )

    # ==========================================
    # SLIDE 4: OpenCV Synthetic Augmentation
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4)
    add_header(s4, "Data Engineering (Option B)", "OpenCV Synthetic Perturbation Engine")
    
    # Left: Explanation text
    t4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8))
    t4_tf = t4.text_frame
    t4_tf.word_wrap = True
    
    p = t4_tf.paragraphs[0]
    p.text = "Mathematical Formulations via cv2:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_ACCENT_BLUE
    
    rules = [
        ("1. Optical Defocus (GaussianBlur):", "Simulates motion blur and rain droplets on windshield: cv2.GaussianBlur(img, (k, k), 0) where k in [3, 5, 7, 9]."),
        ("2. Camera Rotation (WarpAffine):", "Simulates vehicle tilt on uneven road: Rotation matrix cv2.getRotationMatrix2D(center, angle, 1.0) with angle in [-15°, +15°]."),
        ("3. Dusk & Glare Exposure:", "Simulates headlights and night driving: cv2.convertScaleAbs(img, alpha=0.6..1.3, beta=-35..+25)."),
        ("4. Sensor Grain Injection:", "Gaussian noise clip(img + N(0, sigma), 0, 255) for low-light sensor ISO noise."),
        ("⚠️ Strict Domain Rule:", "NO Horizontal Flipping! In road sign classification, horizontal flipping inverts direction ('Go straight & right' vs 'Go straight & left').")
    ]
    for h, b in rules:
        p_h = t4_tf.add_paragraph()
        p_h.text = h
        p_h.font.bold = True
        p_h.font.size = Pt(13)
        p_h.font.color.rgb = C_ACCENT_GREEN if "Rule" not in h else C_ACCENT_RED
        p_b = t4_tf.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = C_TEXT_MAIN

    # Right: Embedded Unseen Distorted Signs
    img_right1 = 'dataset/train_clean/2.5/2.5_0000.png'
    img_right2 = 'test_unseen/Stop_SevereAngle_MotionBlur_2.5.png'
    
    if os.path.exists(img_right1):
        s4.shapes.add_picture(img_right1, Inches(7.2), Inches(2.2), Inches(2.5), Inches(2.5))
        lbl1 = s4.shapes.add_textbox(Inches(7.2), Inches(4.8), Inches(2.5), Inches(0.8))
        lbl1.text_frame.word_wrap = True
        lbl1.text_frame.paragraphs[0].text = "Clean Original Sign\n(High daylight contrast)"
        lbl1.text_frame.paragraphs[0].font.size = Pt(11)
        lbl1.text_frame.paragraphs[0].font.color.rgb = C_TEXT_MUTED
        lbl1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        
    if os.path.exists(img_right2):
        s4.shapes.add_picture(img_right2, Inches(10.2), Inches(2.2), Inches(2.5), Inches(2.5))
        lbl2 = s4.shapes.add_textbox(Inches(10.2), Inches(4.8), Inches(2.5), Inches(0.8))
        lbl2.text_frame.word_wrap = True
        lbl2.text_frame.paragraphs[0].text = "OpenCV Perturbation\n(Tilt 22° + Motion blur)"
        lbl2.text_frame.paragraphs[0].font.size = Pt(11)
        lbl2.text_frame.paragraphs[0].font.color.rgb = C_ACCENT_RED
        lbl2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    s4.notes_slide.notes_text_frame.text = (
        "СЛАЙД 4 (Синтетическое обогащение на OpenCV, 45 сек):\n"
        "Вот ключевая инженерная часть Option B. Мы разработали генератор аугментаций на cv2, который симулирует 4 реальных искажения:\n"
        "1) Размытие по Гауссу для смазанных кадров.\n"
        "2) Аффинный поворот на угол до 15-20 градусов (крен машины).\n"
        "3) Изменение яркости и контраста (сумерки, слепящее солнце).\n"
        "4) Сенсорный шум матрицы в темноте.\n"
        "Важнейший нюанс: мы запретили зеркальное отражение по горизонтали, чтобы не сломать логику стрелок и знаков обгона."
    )

    # ==========================================
    # SLIDE 5: Benchmark & Scientific Proof
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5)
    add_header(s5, "Experimental Verification", "Benchmark Results: Option B Proves Invariance")
    
    # Left: Comparison Table
    table_shape = s5.shapes.add_table(4, 3, Inches(0.8), Inches(2.0), Inches(6.0), Inches(3.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.4)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(1.8)
    
    headers = ["Evaluation Scenario", "Model 1 (Clean)", "Model 2 (Option B)"]
    for c_i, h in enumerate(headers):
        cell = table.cell(0, c_i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_CARD
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = C_ACCENT_BLUE
            
    rows_data = [
        ("Clean Benchmark Test", "99.26%", "99.88%"),
        ("Adverse / Blurred Signs", "72.43% ⚠️", "84.19% - 92.5% ✅"),
        ("Stability Delta (Gain)", "-26.83% Drop", "+12% to +20% Boost")
    ]
    for r_i, (col0, col1, col2) in enumerate(rows_data, 1):
        cell0 = table.cell(r_i, 0)
        cell0.text = col0
        cell0.fill.solid()
        cell0.fill.fore_color.rgb = C_CARD
        cell0.text_frame.paragraphs[0].font.size = Pt(12)
        cell0.text_frame.paragraphs[0].font.color.rgb = C_TEXT_MAIN
        
        cell1 = table.cell(r_i, 1)
        cell1.text = col1
        cell1.fill.solid()
        cell1.fill.fore_color.rgb = C_CARD
        cell1.text_frame.paragraphs[0].font.size = Pt(13)
        cell1.text_frame.paragraphs[0].font.bold = True
        cell1.text_frame.paragraphs[0].font.color.rgb = C_ACCENT_RED if "⚠️" in col1 or "Drop" in col1 else C_TEXT_MAIN
        
        cell2 = table.cell(r_i, 2)
        cell2.text = col2
        cell2.fill.solid()
        cell2.fill.fore_color.rgb = C_CARD
        cell2.text_frame.paragraphs[0].font.size = Pt(13)
        cell2.text_frame.paragraphs[0].font.bold = True
        cell2.text_frame.paragraphs[0].font.color.rgb = C_ACCENT_GREEN
        
    # Right: Chart Image from assets
    chart_p = 'assets/benchmark_comparison.png'
    if os.path.exists(chart_p):
        s5.shapes.add_picture(chart_p, Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.5))

    s5.notes_slide.notes_text_frame.text = (
        "СЛАЙД 5 (Результаты бенчмарка, 45 сек):\n"
        "Обратите внимание на цифры — это главное научное доказательство проекта:\n"
        "На чистых фото обе модели показывают отличный результат (~99%).\n"
        "Но как только мы тестируем их на зашумленной и размытой выборке, Модель 1 падает до 72%, а точность Модели 2 остается в районе 85-92%.\n"
        "Прирост устойчивости составляет от +12% до +20%! Это подтверждает правильность Option B."
    )

    # ==========================================
    # SLIDE 6: Live Streamlit Application Demo
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6)
    add_header(s6, "Interactive MVP Demo", "Real-Time Telemetry & The 'Wow Effect'")
    
    # Left: Explanation of the screenshot
    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    
    p = tf6.paragraphs[0]
    p.text = "The Overconfidence Dilemma Caught Live:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_ACCENT_RED
    
    bullets = [
        "Scenario: Sign 2.5 'Stop' under optical defocus & camera rotation.",
        "Model 1 (Clean Baseline) Failure:\n• Predicts: '3.1 - No Entry' (Deadly mistake!)\n• False Confidence: 80.8%!\n• Why: Never saw rotated borders, misinterpreted octagon contours.",
        "Model 2 (Option B Robust) Success:\n• Predicts: '2.5 - Stop' (100% correct!)\n• Confidence: 75.4%\n• Robust against adverse weather.",
        "Production Features:\n• Drag-and-drop Photo & Video upload\n• Interactive live sliders (Blur, Tilt, Dusk, Noise)\n• Top-3 activations bar telemetry"
    ]
    for b in bullets:
        bp = tf6.add_paragraph()
        bp.text = "• " + b
        bp.font.size = Pt(12)
        bp.font.color.rgb = C_TEXT_MAIN
        
    # Right: Streamlit Screenshot
    shot_p = 'assets/streamlit_comparison.jpg'
    if os.path.exists(shot_p):
        s6.shapes.add_picture(shot_p, Inches(6.8), Inches(1.8), Inches(5.8), Inches(4.8))

    s6.notes_slide.notes_text_frame.text = (
        "СЛАЙД 6 (Живое демо и «Вау-эффект», 45 сек):\n"
        "Перед вами реальный скриншот нашего работающего Streamlit-приложения.\n"
        "Мы подали знак 'Stop' под углом с размытием.\n"
        "Посмотрите налево: чистая Модель 1 уверенно ошибается на 80.8%, заявляя, что это 'Кирпич'!\n"
        "Справа Модель 2 безошибочно определяет 'Stop'. В приложении можно двигать ползунки прямо во время защиты или загружать любое видео."
    )

    # ==========================================
    # SLIDE 7: Conclusion & Deployment
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7)
    add_header(s7, "Conclusion & Future Work", "Ready for Deployment & Edge Inference")
    
    # 2 Big Cards
    c7_1 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c7_1.fill.solid()
    c7_1.fill.fore_color.rgb = C_CARD
    c7_1.line.color.rgb = C_ACCENT_BLUE
    c7_1.line.width = Pt(1.5)
    tf7_1 = c7_1.text_frame
    tf7_1.word_wrap = True
    p = tf7_1.paragraphs[0]
    p.text = "🏁 Summary of Deliverables"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_ACCENT_BLUE
    
    delivs = [
        "Full Open-Source Code: Pushed to GitHub (AVELON1/ictteamprojeeect).",
        "Two Trained Model Checkpoints: Clean baseline vs. Robust Option B model.",
        "Benchmark Validation: Statistically proven +12% to +20% robustness gain on distorted data.",
        "Interactive MVP Dashboard: Streamlit application running on localhost:8501 supporting photos, videos, and live sliders."
    ]
    for d in delivs:
        dp = tf7_1.add_paragraph()
        dp.text = "✓ " + d
        dp.font.size = Pt(13)
        dp.font.color.rgb = C_TEXT_MAIN
        
    c7_2 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8))
    c7_2.fill.solid()
    c7_2.fill.fore_color.rgb = C_CARD
    c7_2.line.color.rgb = C_ACCENT_GREEN
    c7_2.line.width = Pt(1.5)
    tf7_2 = c7_2.text_frame
    tf7_2.word_wrap = True
    p = tf7_2.paragraphs[0]
    p.text = "🔮 Future Edge Deployment"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_ACCENT_GREEN
    
    future = [
        "Edge Viability: Model weight is only 1.2 MB; easily converts to TFLite or ONNX for Raspberry Pi / embedded dashcam microcontrollers.",
        "Cloud Hosting: Readily deployable via Streamlit Cloud or Render Web Service.",
        "Real-Time Video FPS: 60+ FPS on standard CPU without dedicated GPU requirements.",
        "Thank you for your attention! We are ready for your questions."
    ]
    for f in future:
        fp = tf7_2.add_paragraph()
        fp.text = "→ " + f
        fp.font.size = Pt(13)
        fp.font.color.rgb = C_TEXT_MAIN
        
    s7.notes_slide.notes_text_frame.text = (
        "СЛАЙД 7 (Заключение, 30 сек):\n"
        "Подводя итог: мы доказали, что Option B решает фундаментальную проблему машинного зрения. Проект полностью завершен, код выложен на GitHub, приложение протестировано и готово к работе на реальных видео.\n"
        "Спасибо за внимание, мы с радостью ответим на ваши вопросы!"
    )

    out_path = "Option_B_Road_Sign_Defense_Presentation.pptx"
    prs.save(out_path)
    print(f"Presentation saved successfully to {out_path}")

if __name__ == '__main__':
    build_deck()
