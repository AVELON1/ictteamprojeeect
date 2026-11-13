import os
from PIL import Image, ImageDraw, ImageFont

def render_english_card():
    # High-resolution canvas: 1600 x 1300
    W, H = 1600, 1300
    img = Image.new('RGB', (W, H), color=(15, 23, 42)) # Deep navy slate
    draw = ImageDraw.Draw(img)
    
    # Try loading clean system fonts
    font_paths = [
        "/System/Library/Fonts/SFPro-Bold.otf",
        "/System/Library/Fonts/SFProText-Bold.otf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    ]
    font_bold_path = next((p for p in font_paths if os.path.exists(p)), None)
    
    font_reg_paths = [
        "/System/Library/Fonts/SFPro-Regular.otf",
        "/System/Library/Fonts/SFProText-Regular.otf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ]
    font_reg_path = next((p for p in font_reg_paths if os.path.exists(p)), None)
    
    def get_font(bold=False, size=24):
        p = font_bold_path if bold else font_reg_path
        if p:
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
        return ImageFont.load_default()

    f_title = get_font(bold=True, size=46)
    f_sub = get_font(bold=False, size=24)
    f_header = get_font(bold=True, size=22)
    f_badge = get_font(bold=True, size=20)
    f_pred = get_font(bold=True, size=38)
    f_conf = get_font(bold=True, size=52)
    f_conf_sub = get_font(bold=False, size=26)
    f_bar = get_font(bold=False, size=24)

    # 2 Columns
    col_w = 740
    gap = 40
    y_start = 50
    
    # ==========================================
    # LEFT COLUMN: MODEL 1 (CLEAN BASELINE)
    # ==========================================
    x1 = 40
    # Header
    draw.text((x1, y_start), "❌ Model 1 (Clean Baseline)", font=f_title, fill=(244, 63, 94))
    draw.text((x1, y_start + 65), "Trained strictly on pristine clean images", font=f_sub, fill=(148, 163, 184))
    
    # Card Background
    card_y = y_start + 120
    card_h = 360
    draw.rounded_rectangle([x1, card_y, x1 + col_w, card_y + card_h], radius=20, fill=(26, 36, 56), outline=(244, 63, 94), width=3)
    
    # Inside Card
    draw.text((x1 + 35, card_y + 35), "DETECTED CLASS", font=f_header, fill=(148, 163, 184))
    
    # Badge: FALSE PREDICTION
    badge_w, badge_h = 340, 42
    bx = x1 + col_w - badge_w - 35
    by = card_y + 30
    draw.rounded_rectangle([bx, by, bx + badge_w, by + badge_h], radius=10, fill=(76, 29, 39), outline=(244, 63, 94), width=1)
    draw.text((bx + 20, by + 8), "FALSE PREDICTION (MISTAKE)", font=f_badge, fill=(251, 113, 133))
    
    draw.text((x1 + 35, card_y + 105), "3.1 - No entry", font=f_pred, fill=(248, 250, 252))
    draw.text((x1 + 35, card_y + 190), "80.8%", font=f_conf, fill=(244, 63, 94))
    draw.text((x1 + 220, card_y + 210), "confidence (Overconfident!)", font=f_conf_sub, fill=(148, 163, 184))
    
    # Progress bars
    bar_y = card_y + card_h + 45
    draw.text((x1, bar_y), "Top-3 Activations:", font=f_header, fill=(248, 250, 252))
    
    # Bar 1: 3.1 (80.8%)
    draw.text((x1, bar_y + 45), "3.1 - No entry: 80.8%", font=f_bar, fill=(148, 163, 184))
    draw.rounded_rectangle([x1, bar_y + 80, x1 + col_w, bar_y + 102], radius=6, fill=(35, 47, 71))
    draw.rounded_rectangle([x1, bar_y + 80, x1 + int(col_w * 0.808), bar_y + 102], radius=6, fill=(244, 63, 94))
    
    # Bar 2: 2.5 (17.5%)
    draw.text((x1, bar_y + 130), "2.5 - Stop (Ground Truth): 17.5%", font=f_bar, fill=(148, 163, 184))
    draw.rounded_rectangle([x1, bar_y + 165, x1 + col_w, bar_y + 187], radius=6, fill=(35, 47, 71))
    draw.rounded_rectangle([x1, bar_y + 165, x1 + int(col_w * 0.175), bar_y + 187], radius=6, fill=(56, 189, 248))
    
    # Bar 3: 3.20 (0.9%)
    draw.text((x1, bar_y + 215), "3.20 - No overtaking: 0.9%", font=f_bar, fill=(148, 163, 184))
    draw.rounded_rectangle([x1, bar_y + 250, x1 + col_w, bar_y + 272], radius=6, fill=(35, 47, 71))
    draw.rounded_rectangle([x1, bar_y + 250, x1 + max(10, int(col_w * 0.009)), bar_y + 272], radius=6, fill=(100, 116, 139))

    # ==========================================
    # RIGHT COLUMN: MODEL 2 (OPTION B: ROBUST)
    # ==========================================
    x2 = x1 + col_w + gap
    # Header
    draw.text((x2, y_start), "✅ Model 2 (Option B: cv2)", font=f_title, fill=(16, 185, 129))
    draw.text((x2, y_start + 65), "Trained with synthetic cv2 perturbations", font=f_sub, fill=(148, 163, 184))
    
    # Card Background
    draw.rounded_rectangle([x2, card_y, x2 + col_w, card_y + card_h], radius=20, fill=(26, 36, 56), outline=(16, 185, 129), width=3)
    
    # Inside Card
    draw.text((x2 + 35, card_y + 35), "DETECTED CLASS", font=f_header, fill=(148, 163, 184))
    
    # Badge: CORRECT (ROBUST)
    badge_w2, badge_h2 = 250, 42
    bx2 = x2 + col_w - badge_w2 - 35
    by2 = card_y + 30
    draw.rounded_rectangle([bx2, by2, bx2 + badge_w2, by2 + badge_h2], radius=10, fill=(6, 78, 59), outline=(16, 185, 129), width=1)
    draw.text((bx2 + 25, by2 + 8), "CORRECT (ROBUST)", font=f_badge, fill=(52, 211, 153))
    
    draw.text((x2 + 35, card_y + 105), "2.5 - Stop", font=f_pred, fill=(248, 250, 252))
    draw.text((x2 + 35, card_y + 190), "75.4%", font=f_conf, fill=(16, 185, 129))
    draw.text((x2 + 205, card_y + 210), "confidence (Target Verified)", font=f_conf_sub, fill=(148, 163, 184))
    
    # Progress bars
    draw.text((x2, bar_y), "Top-3 Activations:", font=f_header, fill=(248, 250, 252))
    
    # Bar 1: 2.5 (75.4%)
    draw.text((x2, bar_y + 45), "2.5 - Stop (Ground Truth): 75.4%", font=f_bar, fill=(148, 163, 184))
    draw.rounded_rectangle([x2, bar_y + 80, x2 + col_w, bar_y + 102], radius=6, fill=(35, 47, 71))
    draw.rounded_rectangle([x2, bar_y + 80, x2 + int(col_w * 0.754), bar_y + 102], radius=6, fill=(16, 185, 129))
    
    # Bar 2: 3.1 (9.7%)
    draw.text((x2, bar_y + 130), "3.1 - No entry: 9.7%", font=f_bar, fill=(148, 163, 184))
    draw.rounded_rectangle([x2, bar_y + 165, x2 + col_w, bar_y + 187], radius=6, fill=(35, 47, 71))
    draw.rounded_rectangle([x2, bar_y + 165, x2 + int(col_w * 0.097), bar_y + 187], radius=6, fill=(100, 116, 139))
    
    # Bar 3: 3.27 (4.8%)
    draw.text((x2, bar_y + 215), "3.27 - No stopping: 4.8%", font=f_bar, fill=(148, 163, 184))
    draw.rounded_rectangle([x2, bar_y + 250, x2 + col_w, bar_y + 272], radius=6, fill=(35, 47, 71))
    draw.rounded_rectangle([x2, bar_y + 250, x2 + int(col_w * 0.048), bar_y + 272], radius=6, fill=(100, 116, 139))

    out_p = 'assets/streamlit_en_clean.png'
    img.save(out_p, quality=95)
    print(f"Generated clean English telemetry card: {out_p}")

if __name__ == '__main__':
    render_english_card()
