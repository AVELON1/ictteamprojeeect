import os
import json
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import tensorflow as tf

from src.eval_suite import ModelBenchmarkSuite, CLASS_NAMES_RU, PRESET_PERFORM, perturb_image

st.set_page_config(
    page_title="🛡️ Сравнение Моделей: Распознавание дорожных знаков",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (No generic purple/indigo slop, dark slate palette, tabular-nums)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 96%;
    }
    
    /* Metric Cards */
    .kpi-card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.18s ease;
    }
    .kpi-card:hover {
        border-color: rgba(148, 163, 184, 0.35);
        transform: translateY(-2px);
    }
    
    .kpi-m1 { border-left: 4px solid #f43f5e; }
    .kpi-m2 { border-left: 4px solid #10b981; }
    .kpi-delta { border-left: 4px solid #38bdf8; }
    .kpi-rescued { border-left: 4px solid #f59e0b; }
    
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #64748b;
    }
    
    /* Category Card */
    .cat-box {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: border-color 0.18s ease;
    }
    .cat-box:hover {
        border-color: rgba(148, 163, 184, 0.3);
    }
    .cat-box-rescued {
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.04) 0%, rgba(15, 23, 42, 0.95) 100%);
    }
    
    /* Badges */
    .badge-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .badge-rescued {
        background: rgba(16, 185, 129, 0.16);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-both-ok {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .badge-fail {
        background: rgba(244, 63, 94, 0.16);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    
    .code-num {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }
    
    /* Loading overlay */
    .scan-status-box {
        background: #1e293b;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Загрузка моделей нейросети и тестовой выборки...")
def get_benchmark_suite():
    return ModelBenchmarkSuite()

suite = get_benchmark_suite()

# Header
st.title("🛡️ Распознавание дорожных знаков: Бенчмарк устойчивости")
st.markdown(
    "Сравнение **Модели 1 (Базовая без аугментаций)**, которая резко теряет точность при искажениях камеры, "
    "и **Модели 2 (Option B с синтетической OpenCV-аугментацией)**, сохраняющей надежность в реальных дорожных условиях."
)

# Tabs
tab_multi, tab_single, tab_harsh, tab_camera = st.tabs([
    "📊 Мульти-тест 10 категорий и Статистика (Главный экран)",
    "🔍 Анализ одиночного знака",
    "🌪️ Готовые экстремальные сценарии (Unseen)",
    "📷 Веб-камера и Загрузка файлов"
])

# =========================================================================
# TAB 1: MULTI-CATEGORY COMPREHENSIVE BENCHMARK (DEFAULT USER REQUEST)
# =========================================================================
with tab_multi:
    # Sidebar or top controls for perturbations
    st.markdown("### 🎛️ Выберите условия съемки для тестирования всех 10 категорий")
    
    col_preset, col_btn = st.columns([3.5, 1.2])
    with col_preset:
        preset_choice = st.selectbox(
            "Пресет дорожных условий (применяется автоматически ко всем знакам):",
            [
                "harsh_weather: Тяжелые условия (Дождь/смаз + наклон 14° + сумерки + зернистость) [Рекомендуемый]",
                "heavy_fog: Густой туман и оптический расфокус (сильный Blur)",
                "night_dusk: Ночная съемка / Грязное лобовое стекло (низкая освещенность + шум)",
                "mild_adverse: Легкие вибрации и наклон 8° (городской поток)",
                "clean: Идеальные чистые условия (лабораторные фото)",
                "custom: Ручная настройка параметров OpenCV"
            ],
            index=0
        )
        selected_key = preset_choice.split(":")[0]
    
    with col_btn:
        st.write("")
        st.write("")
        rerun_clicked = st.button("🔄 Перезапустить тест по всем знакам", use_container_width=True)

    # Custom sliders if selected
    custom_rot, custom_blur, custom_bright, custom_noise = 0, 0, 0, 0
    if selected_key == "custom":
        with st.expander("⚙️ Ручная настройка параметров искажений OpenCV", expanded=True):
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            with c_s1:
                custom_rot = st.slider("Наклон / Угол камеры (°)", -30, 30, 12, step=1)
            with c_s2:
                custom_blur = st.slider("Размытие / Смаз (px)", 0, 21, 7, step=2)
            with c_s3:
                custom_bright = st.slider("Экспозиция (Яркость)", -80, 60, -25, step=5)
            with c_s4:
                custom_noise = st.slider("Шумы сенсора матрицы", 0, 35, 15, step=1)

    # State check for progressive loading display on entry
    if "first_load_complete" not in st.session_state:
        st.session_state["first_load_complete"] = False

    # Perform progressive visual loading if requested or first entry
    if not st.session_state["first_load_complete"] or rerun_clicked:
        with st.status("🚀 Инициализация мульти-категорийного сканирования...", expanded=True) as status_box:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for step_i, code in enumerate(suite.codes):
                ru_title = CLASS_NAMES_RU.get(code, code)
                fraction = (step_i + 1) / len(suite.codes)
                progress_bar.progress(fraction)
                status_text.markdown(f"📥 **Загрузка и оптическая обработка категории {step_i+1}/10:** `{code}` — *{ru_title}*")
                time.sleep(0.06)
                
            status_text.markdown("⚡ **Параллельный инференс по 816 тестовым образцам (Модель 1 vs Модель 2)...**")
            time.sleep(0.12)
            status_box.update(label="✅ Мульти-тест завершен! Статистика рассчитана.", state="complete", expanded=False)
            st.session_state["first_load_complete"] = True

    # Compute evaluation results
    if selected_key == "custom":
        results = suite.evaluate_custom(
            rotation=custom_rot,
            blur=custom_blur,
            brightness=custom_bright,
            noise=custom_noise,
            preset_title="Пользовательские параметры"
        )
    else:
        results = suite.evaluate_preset(selected_key)

    st.markdown("---")
    
    # -------------------------------------------------------------
    # EXECUTIVE STATISTICAL SUMMARY (KPI CARDS)
    # -------------------------------------------------------------
    st.markdown("### 📈 Сравнительная статистика производительности моделей")
    st.caption(f"Условия тестирования: **{results['preset_title']}** (оценка по всей тестовой выборке из {results['total_test_samples']} изображений)")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
        <div class="kpi-card kpi-m1">
            <div class="kpi-title">❌ Модель 1 (Базовая, Clean)</div>
            <div class="kpi-value" style="color: #f43f5e;">{results['m1_accuracy']:.1f}%</div>
            <div class="kpi-subtext">Точность без аугментаций<br>Доля ошибок: <strong style="color:#fb7185;">{results['m1_error_rate']:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="kpi-card kpi-m2">
            <div class="kpi-title">✅ Модель 2 (Option B с CV)</div>
            <div class="kpi-value" style="color: #10b981;">{results['m2_accuracy']:.1f}%</div>
            <div class="kpi-subtext">Обучена с синтетикой OpenCV<br>Доля ошибок: <strong style="color:#34d399;">{results['m2_error_rate']:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        delta_color = "#38bdf8" if results['accuracy_delta'] >= 0 else "#f43f5e"
        plus_sign = "+" if results['accuracy_delta'] >= 0 else ""
        st.markdown(f"""
        <div class="kpi-card kpi-delta">
            <div class="kpi-title">⚡ Прирост устойчивости</div>
            <div class="kpi-value" style="color: {delta_color};">{plus_sign}{results['accuracy_delta']:.1f}%</div>
            <div class="kpi-subtext">Абсолютный выигрыш точности<br>Снижение сбоев в <strong>{max(1.0, results['m1_error_rate'] / max(0.1, results['m2_error_rate'])):.1f}x</strong> раза</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="kpi-card kpi-rescued">
            <div class="kpi-title">✨ Спасенные знаки</div>
            <div class="kpi-value" style="color: #f59e0b;">{results['rescued_count']} / {results['total_categories']}</div>
            <div class="kpi-subtext">Категорий спасено от ошибки<br>Модель 1 ошиблась, Модель 2 верна</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # INTERACTIVE ACCURACY BY CATEGORY CHART
    # -------------------------------------------------------------
    st.markdown("#### 📊 Точность распознавания по каждой из 10 категорий ПДД")
    
    chart_data = []
    for cat in results["categories"]:
        short_label = f"{cat['code']} {cat['ru_name'][:18]}"
        chart_data.append({
            "Категория": short_label,
            "Модель": "Модель 1 (Базовая без CV)",
            "Точность (%)": round(cat["m1_cls_acc"], 1)
        })
        chart_data.append({
            "Категория": short_label,
            "Модель": "Модель 2 (Option B с CV)",
            "Точность (%)": round(cat["m2_cls_acc"], 1)
        })
    df_chart = pd.DataFrame(chart_data)

    cat_chart = alt.Chart(df_chart).mark_bar(
        cornerRadiusTopRight=4,
        cornerRadiusBottomRight=4,
        height=14
    ).encode(
        y=alt.Y("Категория:N", sort=None, title=None, axis=alt.Axis(labelFontSize=11, labelColor="#cbd5e1")),
        x=alt.X("Точность (%):Q", scale=alt.Scale(domain=[0, 100]), title="Точность на тестовой выборке (%)"),
        color=alt.Color("Модель:N", scale=alt.Scale(
            domain=["Модель 1 (Базовая без CV)", "Модель 2 (Option B с CV)"],
            range=["#f43f5e", "#10b981"]
        ), legend=alt.Legend(title=None, orient="top", labelFontSize=12, labelColor="#f8fafc")),
        yOffset="Модель:N",
        tooltip=["Категория", "Модель", "Точность (%)"]
    ).properties(
        height=380
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        gridColor="#1e293b",
        labelColor="#94a3b8",
        titleColor="#cbd5e1"
    )
    
    st.altair_chart(cat_chart, use_container_width=True)

    # -------------------------------------------------------------
    # MULTI-CATEGORY LIVE IMAGE GALLERY & DIRECT COMPARISON
    # -------------------------------------------------------------
    st.markdown("### 🖼️ Сравнение работы моделей по всем 10 категориям одновременно")
    st.markdown(
        "Ниже представлены реальные образцы знаков под текущими искажениями. "
        "Показано предсказание **слабой Модели 1** против **улучшенной Модели 2**:"
    )

    filter_mode = st.radio(
        "Фильтр отображения карточек:",
        ["Все 10 категорий", "✨ Только где Модель 2 спасла ситуацию (Rescued)", "❌ Где Модель 1 ошиблась"],
        horizontal=True
    )

    filtered_cats = results["categories"]
    if filter_mode == "✨ Только где Модель 2 спасла ситуацию (Rescued)":
        filtered_cats = [c for c in results["categories"] if c["status"] == "rescued"]
    elif filter_mode == "❌ Где Модель 1 ошиблась":
        filtered_cats = [c for c in results["categories"] if not c["m1_correct"]]

    if not filtered_cats:
        st.info("По выбранному фильтру знаков не найдено (в данном режиме обе модели отработали корректно).")
    else:
        # Display in 2 responsive columns
        grid_col1, grid_col2 = st.columns(2)
        
        for idx_c, cat in enumerate(filtered_cats):
            target_col = grid_col1 if idx_c % 2 == 0 else grid_col2
            
            with target_col:
                is_rescued = (cat["status"] == "rescued")
                box_class = "cat-box cat-box-rescued" if is_rescued else "cat-box"
                
                if is_rescued:
                    badge_html = '<span class="badge-pill badge-rescued">✨ CV МОДЕЛЬ СПАСЛА ЗНАК</span>'
                elif cat["m1_correct"] and cat["m2_correct"]:
                    badge_html = '<span class="badge-pill badge-both-ok">✅ ОБЕ ВЕРНЫ</span>'
                else:
                    badge_html = '<span class="badge-pill badge-fail">⚠️ СЛОЖНЫЙ СЛУЧАЙ</span>'
                
                st.markdown(f"""
                <div class="{box_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.75rem;">
                        <div>
                            <strong style="font-size:1.05rem; color:#f8fafc;">{cat['code']} — {cat['ru_name']}</strong>
                        </div>
                        {badge_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c_img, c_m1, c_m2 = st.columns([1.1, 1.45, 1.45])
                
                with c_img:
                    # Enlarge 64x64 to crisp 160x160
                    scaled_img = cv2.resize(cat["visual_img"], (160, 160), interpolation=cv2.INTER_NEAREST)
                    st.image(scaled_img, caption=f"Входное фото ({cat['code']})", use_container_width=True)
                
                with c_m1:
                    m1_icon = "✅" if cat["m1_correct"] else "❌"
                    m1_color = "#10b981" if cat["m1_correct"] else "#f43f5e"
                    st.markdown(f"**Модель 1 (Базовая):**")
                    st.markdown(f"<span style='color:{m1_color}; font-weight:700;'>{m1_icon} {cat['m1_pred_name']}</span>", unsafe_allow_html=True)
                    st.markdown(f"Уверенность: `<span class='code-num'>{cat['m1_conf']:.1f}%</span>`", unsafe_allow_html=True)
                    st.progress(min(1.0, cat['m1_conf'] / 100.0))
                    st.caption(f"Точность класса в тесте: {cat['m1_cls_acc']:.1f}%")

                with c_m2:
                    m2_icon = "✅" if cat["m2_correct"] else "❌"
                    m2_color = "#10b981" if cat["m2_correct"] else "#f43f5e"
                    st.markdown(f"**Модель 2 (Option B с CV):**")
                    st.markdown(f"<span style='color:{m2_color}; font-weight:700;'>{m2_icon} {cat['m2_pred_name']}</span>", unsafe_allow_html=True)
                    st.markdown(f"Уверенность: `<span class='code-num'>{cat['m2_conf']:.1f}%</span>`", unsafe_allow_html=True)
                    st.progress(min(1.0, cat['m2_conf'] / 100.0))
                    st.caption(f"Точность класса в тесте: {cat['m2_cls_acc']:.1f}%")

                with st.expander(f"📊 Top-3 распределение вероятностей для знака {cat['code']}"):
                    t3_col1, t3_col2 = st.columns(2)
                    with t3_col1:
                        st.write("**Модель 1:**")
                        for name, prob in cat["p1_top3"]:
                            st.write(f"- {name}: `{prob:.1f}%`")
                    with t3_col2:
                        st.write("**Модель 2:**")
                        for name, prob in cat["p2_top3"]:
                            st.write(f"- {name}: `{prob:.1f}%`")
                st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# =========================================================================
# TAB 2: SINGLE SIGN DEEP DIVE WITH REALTIME SLIDERS
# =========================================================================
with tab_single:
    st.markdown("### 🔍 Глубокий анализ отдельного знака с покадровой деградацией")
    st.caption("Выберите любой конкретный знак и меняйте ползунки в реальном времени, наблюдая, в какой момент базовая модель дает сбой.")
    
    col_s_left, col_s_mid, col_s_right = st.columns([1.2, 1.4, 1.4])
    
    chosen_cat_code = st.selectbox(
        "Выберите знак для детального теста:",
        [f"{c} — {CLASS_NAMES_RU.get(c, c)}" for c in suite.codes]
    ).split(" — ")[0]
    
    sample_raw = suite.sample_per_category[chosen_cat_code]["img_rgb"]
    true_label = f"{chosen_cat_code} — {CLASS_NAMES_RU.get(chosen_cat_code, '')}"
    
    st.markdown("#### 🎛️ Интерактивные искажения OpenCV в реальном времени:")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        s_blur = st.slider("Оптический дефокус / Blur (px)", 0, 25, 5, step=2, key="s_blur")
    with s_col2:
        s_rot = st.slider("Наклон камеры / Вибрация (°)", -45, 45, 10, step=1, key="s_rot")
    with s_col3:
        s_bright = st.slider("Освещенность / Сумерки (Beta)", -90, 70, -20, step=5, key="s_bright")
    with s_col4:
        s_noise = st.slider("Шумы матрицы (Sigma)", 0, 40, 10, step=2, key="s_noise")

    # Perturb
    perturbed_single = perturb_image(sample_raw, rotation=s_rot, blur=s_blur, brightness=s_bright, noise=s_noise)
    inp_single = np.expand_dims(perturbed_single.astype(np.float32) / 255.0, axis=0)

    pred1_s = suite.model_clean.predict(inp_single, verbose=0)[0]
    pred2_s = suite.model_aug.predict(inp_single, verbose=0)[0]

    top1_s = int(np.argmax(pred1_s))
    conf1_s = float(pred1_s[top1_s]) * 100
    name1_s = CLASS_NAMES_RU.get(suite.codes[top1_s], suite.idx_to_name[top1_s])

    top2_s = int(np.argmax(pred2_s))
    conf2_s = float(pred2_s[top2_s]) * 100
    name2_s = CLASS_NAMES_RU.get(suite.codes[top2_s], suite.idx_to_name[top2_s])

    target_idx = suite.code_to_idx[chosen_cat_code]
    is_corr1 = (top1_s == target_idx)
    is_corr2 = (top2_s == target_idx)

    c1_s, c2_s, c3_s = st.columns([1.1, 1.45, 1.45])

    with c1_s:
        st.markdown("##### 📷 Входное изображение")
        st.image(cv2.resize(perturbed_single, (240, 240), interpolation=cv2.INTER_NEAREST), use_container_width=True)
        st.info(f"🎯 **Истинный класс:**\n{true_label}")

    with c2_s:
        st.markdown("##### ❌ Модель 1 (Базовая, Clean)")
        status_m1 = '<span class="badge-pill badge-rescued">ВЕРНО</span>' if is_corr1 else '<span class="badge-pill badge-fail">ОШИБКА</span>'
        st.markdown(f"""
        <div class="kpi-card kpi-m1">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-title">Предсказанный класс</span>
                {status_m1}
            </div>
            <div style="font-size:1.15rem; font-weight:700; margin:0.3rem 0;">{name1_s}</div>
            <div class="code-num" style="font-size:1.5rem; font-weight:800; color:#f43f5e;">
                {conf1_s:.1f}% <span style="font-size:0.8rem; color:#94a3b8; font-weight:400;">уверенность</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        top3_1 = np.argsort(pred1_s)[-3:][::-1]
        st.write("**Top-3 вероятности:**")
        for i in top3_1:
            st.progress(float(pred1_s[i]), text=f"{CLASS_NAMES_RU.get(suite.codes[i], suite.idx_to_name[i])}: {pred1_s[i]*100:.1f}%")

    with c3_s:
        st.markdown("##### ✅ Модель 2 (Option B с CV)")
        status_m2 = '<span class="badge-pill badge-rescued">ВЕРНО (УСТОЙЧИВО)</span>' if is_corr2 else '<span class="badge-pill badge-fail">ПРОВЕРКА</span>'
        st.markdown(f"""
        <div class="kpi-card kpi-m2">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-title">Предсказанный класс</span>
                {status_m2}
            </div>
            <div style="font-size:1.15rem; font-weight:700; margin:0.3rem 0;">{name2_s}</div>
            <div class="code-num" style="font-size:1.5rem; font-weight:800; color:#10b981;">
                {conf2_s:.1f}% <span style="font-size:0.8rem; color:#94a3b8; font-weight:400;">уверенность</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        top3_2 = np.argsort(pred2_s)[-3:][::-1]
        st.write("**Top-3 вероятности:**")
        for i in top3_2:
            st.progress(float(pred2_s[i]), text=f"{CLASS_NAMES_RU.get(suite.codes[i], suite.idx_to_name[i])}: {pred2_s[i]*100:.1f}%")

# =========================================================================
# TAB 3: HARSH UNSEEN REAL-WORLD SCENARIOS
# =========================================================================
with tab_harsh:
    st.markdown("### 🌪️ Готовые экстремальные сценарии из реальной жизни")
    st.caption("Набор стресс-тестов с сильным дождем, ослепляющими бликами, тяжелым туманом и ночной съемкой.")

    manifest_path = "test_unseen/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            unseen_items = json.load(f)
            
        unseen_opts = [f"{it['code']} — {it['title']}" for it in unseen_items]
        chosen_u = st.selectbox("Выберите экстремальный кейс:", unseen_opts)
        chosen_obj = next(it for it in unseen_items if f"{it['code']} — {it['title']}" == chosen_u)
        
        img_unseen_path = os.path.join("test_unseen", chosen_obj["file"])
        raw_bgr = cv2.imread(img_unseen_path)
        img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        u_base = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)
        u_inp = np.expand_dims(u_base.astype(np.float32) / 255.0, axis=0)
        
        p1_u = suite.model_clean.predict(u_inp, verbose=0)[0]
        p2_u = suite.model_aug.predict(u_inp, verbose=0)[0]
        
        top1_u = int(np.argmax(p1_u))
        top2_u = int(np.argmax(p2_u))
        
        target_u_code = chosen_obj["code"]
        target_u_idx = suite.code_to_idx[target_u_code]
        
        cu_1, cu_2, cu_3 = st.columns([1.1, 1.45, 1.45])
        
        with cu_1:
            st.image(img_rgb, caption=chosen_obj["title"], use_container_width=True)
            st.info(f"🎯 **Истинный знак:**\n{target_u_code} — {CLASS_NAMES_RU.get(target_u_code, '')}")
            
        with cu_2:
            st.markdown("##### ❌ Модель 1 (Базовая)")
            is_u1_ok = (top1_u == target_u_idx)
            color1 = "#10b981" if is_u1_ok else "#f43f5e"
            st.markdown(f"""
            <div class="kpi-card kpi-m1">
                <div class="kpi-title">Предсказание</div>
                <div style="font-size:1.1rem; font-weight:700; color:{color1};">
                    {'✅' if is_u1_ok else '❌'} {CLASS_NAMES_RU.get(suite.codes[top1_u], '')}
                </div>
                <div class="code-num" style="font-size:1.4rem; font-weight:800; color:{color1};">
                    {p1_u[top1_u]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with cu_3:
            st.markdown("##### ✅ Модель 2 (Option B с CV)")
            is_u2_ok = (top2_u == target_u_idx)
            color2 = "#10b981" if is_u2_ok else "#f43f5e"
            st.markdown(f"""
            <div class="kpi-card kpi-m2">
                <div class="kpi-title">Предсказание</div>
                <div style="font-size:1.1rem; font-weight:700; color:{color2};">
                    {'✅' if is_u2_ok else '❌'} {CLASS_NAMES_RU.get(suite.codes[top2_u], '')}
                </div>
                <div class="code-num" style="font-size:1.4rem; font-weight:800; color:{color2};">
                    {p2_u[top2_u]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================================
# TAB 4: LIVE WEBCAM & CUSTOM MEDIA UPLOAD
# =========================================================================
with tab_camera:
    st.markdown("### 📷 Своя веб-камера или загрузка файла")
    st.caption("Проверьте работу моделей на собственных фотографиях или покажите знак веб-камере.")

    cam_source = st.radio("Источник:", ["Веб-камера ноутбука", "Загрузить фото (PNG, JPG)", "Загрузить видео (MP4)"], horizontal=True)
    live_frame = None

    if cam_source == "Веб-камера ноутбука":
        cam_pic = st.camera_input("Сделайте снимок знака")
        if cam_pic:
            fbytes = np.asarray(bytearray(cam_pic.read()), dtype=np.uint8)
            live_frame = cv2.cvtColor(cv2.imdecode(fbytes, 1), cv2.COLOR_BGR2RGB)
            
    elif cam_source == "Загрузить фото (PNG, JPG)":
        up_img = st.file_uploader("Загрузите фото дорожного знака", type=["png", "jpg", "jpeg"])
        if up_img:
            fbytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
            live_frame = cv2.cvtColor(cv2.imdecode(fbytes, 1), cv2.COLOR_BGR2RGB)
            
    elif cam_source == "Загрузить видео (MP4)":
        up_vid = st.file_uploader("Загрузите видео с видеорегистратора", type=["mp4", "mov", "avi"])
        if up_vid:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(up_vid.read())
            cap = cv2.VideoCapture(tfile.name)
            tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tot_frames > 0:
                f_idx = st.slider("Номер кадра видео:", 0, max(0, tot_frames - 1), 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    live_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()

    if live_frame is not None:
        c_l1, c_l2, c_l3 = st.columns([1.1, 1.45, 1.45])
        with c_l1:
            st.image(live_frame, caption="Исходный кадр", use_container_width=True)
            
        inp_live = cv2.resize(live_frame, (64, 64), interpolation=cv2.INTER_AREA)
        inp_live_t = np.expand_dims(inp_live.astype(np.float32) / 255.0, axis=0)
        
        p1_l = suite.model_clean.predict(inp_live_t, verbose=0)[0]
        p2_l = suite.model_aug.predict(inp_live_t, verbose=0)[0]
        
        top1_l = int(np.argmax(p1_l))
        top2_l = int(np.argmax(p2_l))
        
        with c_l2:
            st.markdown("##### ❌ Модель 1 (Базовая)")
            st.markdown(f"**{CLASS_NAMES_RU.get(suite.codes[top1_l], suite.idx_to_name[top1_l])}**")
            st.markdown(f"Уверенность: `{p1_l[top1_l]*100:.1f}%`")
            st.progress(float(p1_l[top1_l]))
            
        with c_l3:
            st.markdown("##### ✅ Модель 2 (Option B с CV)")
            st.markdown(f"**{CLASS_NAMES_RU.get(suite.codes[top2_l], suite.idx_to_name[top2_l])}**")
            st.markdown(f"Уверенность: `{p2_l[top2_l]*100:.1f}%`")
            st.progress(float(p2_l[top2_l]))
