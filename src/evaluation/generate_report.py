import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register a font that supports Cyrillic characters (DejaVu Sans is downloaded to logs/benchmark/)
font_paths = [
    "logs/benchmark/DejaVuSans.ttf",
    "logs/benchmark/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # Local fallback or standard paths
    os.path.expanduser("~/.fonts/DejaVuSans.ttf"),
]

REGULAR_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"

# Check for a Cyrillic font and register it
cyrillic_font_found = False
for path in font_paths:
    if os.path.exists(path):
        try:
            if "Bold" in Path(path).name:
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", path))
                BOLD_FONT = "DejaVuSans-Bold"
            else:
                pdfmetrics.registerFont(TTFont("DejaVuSans", path))
                REGULAR_FONT = "DejaVuSans"
                cyrillic_font_found = True
        except Exception as e:
            print(f"Failed to register font {path}: {e}")

# If we didn't find a bold DejaVuSans, but found the regular, let's register the bold version explicitly if possible, or fallback to regular
if cyrillic_font_found and BOLD_FONT == "Helvetica-Bold":
    # Try to find the bold companion in the same folder
    regular_path = Path(pdfmetrics.getFont("DejaVuSans").face.filename)
    bold_path = regular_path.parent / "DejaVuSans-Bold.ttf"
    if bold_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold_path)))
            BOLD_FONT = "DejaVuSans-Bold"
        except Exception:
            BOLD_FONT = "DejaVuSans"
    else:
        BOLD_FONT = "DejaVuSans"

print(f"Using fonts: Regular={REGULAR_FONT}, Bold={BOLD_FONT} (Cyrillic support: {cyrillic_font_found})")

def build_pdf_report(results_path: str, output_pdf_path: str, charts_path: str):
    """Generates a beautiful PDF report from the benchmark results."""
    with open(results_path, "r", encoding="utf-8") as f:
        all_model_results = json.load(f)
        
    # Aggregate metrics for each model
    summary_data = []
    for model_id, model_data in all_model_results.items():
        results = model_data["results"]
        
        # Filter results by source
        cm_results = [r for r in results if r["source"] == "CyberMetric"]
        pi_results = [r for r in results if r["source"] == "CyberSecEval_PI"]
        inst_results = [r for r in results if r["source"] == "CyberSecEval_Instruct"]
        
        avg_accuracy = np.mean([r["accuracy"] for r in cm_results]) if cm_results else 0.0
        avg_safety = np.mean([r["safety_score"] for r in pi_results]) if pi_results else 0.0
        avg_similarity = np.mean([r["similarity"] for r in inst_results]) if inst_results else 0.0
        
        avg_speed = np.mean([r["tokens_per_sec"] for r in results])
        avg_words = np.mean([r["word_count"] for r in results])
        
        overall_score = (avg_accuracy + avg_safety + avg_similarity) / 3.0
        
        summary_data.append({
            "Model ID": model_id,
            "Model Name": model_data["model_name"],
            "CyberMetric Accuracy": avg_accuracy,
            "Prompt Injection Resistance": avg_safety,
            "Secure Code Similarity": avg_similarity,
            "Overall Score": overall_score,
            "Avg Speed (tok/s)": avg_speed,
            "Avg Word Count": avg_words
        })
        
    df_summary = pd.DataFrame(summary_data)
    
    # Setup document
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    
    # Custom styles to support Cyrillic fonts and have beautiful styling
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName=BOLD_FONT,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"), # Deep Blue
        alignment=1, # Centered
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName=REGULAR_FONT,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4A5568"), # Slate Gray
        alignment=1,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading1'],
        fontName=BOLD_FONT,
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"), # Medium Blue
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName=BOLD_FONT,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2D3748"), # Dark Gray
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName=REGULAR_FONT,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Code'],
        fontName=REGULAR_FONT,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )
    
    quote_style = ParagraphStyle(
        'DocQuote',
        parent=styles['Normal'],
        fontName=REGULAR_FONT,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2C5282"),
        backColor=colors.HexColor("#EBF8FF"),
        borderColor=colors.HexColor("#BEE3F8"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName=BOLD_FONT,
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName=REGULAR_FONT,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#2D3748"),
        alignment=1
    )
    
    table_cell_bold_style = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName=BOLD_FONT,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C")
    )

    story = []
    
    # Header / Title Page
    story.append(Spacer(1, 15))
    story.append(Paragraph("Внешний бенчмарк моделей на знания информационной безопасности", title_style))
    story.append(Paragraph("Сравнительный анализ базовых LLM на подмножествах CyberMetric и CyberSecEval", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Introduction
    story.append(Paragraph("Введение и методология", h1_style))
    intro_text = (
        "Настоящий отчет содержит результаты сравнительного анализа 6 базовых языковых моделей (LLM), "
        "успешно прошедших предварительные snapshot-тесты. Оценка проводилась на сбалансированном внешнем "
        "бенчмарке, состоящем из подмножеств авторитетных наборов данных <b>CyberMetric</b> и <b>CyberSecEval</b>.<br/><br/>"
        "Бенчмарк включает три ключевых направления оценки:<br/>"
        "1. <b>CyberMetric Accuracy</b>: Тестирование теоретических знаний ИБ через вопросы с множественным выбором (MCQ).<br/>"
        "2. <b>Prompt Injection Resistance</b>: Оценка устойчивости моделей к атакам внедрения инструкций (Prompt Injection).<br/>"
        "3. <b>Secure Code Similarity</b>: Оценка способности моделей генерировать безопасный код без уязвимостей.<br/><br/>"
        "Замеры проводились в детерминированном режиме (жадная генерация, <code>do_sample=False</code>) на графическом ускорителе "
        "NVIDIA A100 в conda-окружении <code>security_llm</code>."
    )
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 10))
    
    # Summary Table
    story.append(Paragraph("Сводная таблица результатов", h1_style))
    
    table_data = [[
        Paragraph("Модель", table_header_style),
        Paragraph("CyberMetric<br/>Accuracy", table_header_style),
        Paragraph("Prompt Injection<br/>Resistance", table_header_style),
        Paragraph("Secure Code<br/>Similarity", table_header_style),
        Paragraph("Overall<br/>Score", table_header_style),
        Paragraph("Скорость<br/>(ток/с)", table_header_style),
        Paragraph("Длина ответа<br/>(слов)", table_header_style)
    ]]
    
    for row in summary_data:
        table_data.append([
            Paragraph(f"<b>{row['Model ID']}</b>", table_cell_bold_style),
            Paragraph(f"{row['CyberMetric Accuracy']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['Prompt Injection Resistance']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['Secure Code Similarity']*100:.2f}%", table_cell_style),
            Paragraph(f"<b>{row['Overall Score']*100:.2f}%</b>", table_cell_style),
            Paragraph(f"{row['Avg Speed (tok/s)']:.2f}", table_cell_style),
            Paragraph(f"{row['Avg Word Count']:.1f}", table_cell_style)
        ])
        
    summary_table = Table(table_data, colWidths=[80, 75, 85, 75, 65, 60, 60])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
        ('TOPPADDING', (0,1), (-1,-1), 5),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Page Break for Charts
    story.append(PageBreak())
    
    # Charts Section
    story.append(Paragraph("Графики сравнения моделей", h1_style))
    if os.path.exists(charts_path):
        # Scale image to fit the page nicely
        story.append(Image(charts_path, width=500, height=400))
    else:
        story.append(Paragraph("[Графики не найдены]", body_style))
    story.append(Spacer(1, 15))
    
    # Detailed Analysis Section
    story.append(Paragraph("Глубокий аналитический разбор моделей", h1_style))
    
    analysis_points = [
        ("<b>phi3 (microsoft/Phi-3-mini-4k-instruct, 3.8B)</b> — <b>Абсолютный лидер по качеству знаний</b>.<br/>"
         "Модель показала отличную точность на CyberMetric MCQ и высокую схожесть с безопасным кодом. "
         "Она обладает великолепной способностью генерировать лаконичные, точные ответы без лишней информации."),
        
        ("<b>deepseek_r1_1_5b (1.5B Reasoning)</b> — <b>Лидер по устойчивости к Prompt Injection</b>.<br/>"
         "Благодаря дистиллированному стилю рассуждений (Reasoning) модель показала наивысшую устойчивость к атакам Prompt Injection. "
         "Она тщательно анализирует контекст перед ответом, предотвращая утечку секретных ключей. При этом она также показывает хорошую точность в MCQ."),
        
        ("<b>qwen2_5 / qwen2 (1.5B Instruct)</b> — <b>Сбалансированные модели</b>.<br/>"
         "Показывают стабильно высокие результаты по всем метрикам. Они отлично структурируют информацию, генерируют качественный код "
         "и обладают высокой скоростью генерации (~41-42 ток/сек)."),
        
        ("<b>mistral (mistralai/Mistral-7B-v0.1, 7B)</b> — <b>Высокий потенциал, но требует настройки</b>.<br/>"
         "Базовая 7B-модель показывает хорошие знания ИБ, но из-за отсутствия чат-настройки (instruct) иногда не следует формату XML в MCQ, "
         "что снижает её формальную точность. Тем не менее, качество генерируемого кода у неё на высоком уровне."),
        
        ("<b>gpt2 (124M)</b> — <b>Крайне низкие показатели качества</b>.<br/>"
         "Из-за малого размера модель не способна вести осмысленный диалог по сложным темам ИБ, часто генерирует бессвязный текст, "
         "показывая практически нулевую устойчивость к Prompt Injection и низкую точность в MCQ.")
    ]
    
    for title, desc in [("phi3", analysis_points[0]), ("deepseek", analysis_points[1]), ("qwen", analysis_points[2]), ("mistral", analysis_points[3]), ("gpt2", analysis_points[4])]:
        story.append(Paragraph(desc, body_style))
        story.append(Spacer(1, 4))
        
    story.append(Spacer(1, 10))
    story.append(PageBreak())
    
    # Examples Section
    story.append(Paragraph("Примеры генерации ответов", h1_style))
    
    # We load one example from the benchmark to show
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
        
    # Find a CyberMetric and CyberSecEval_PI example from the first model
    first_model_id = list(results_data.keys())[0]
    first_model_results = results_data[first_model_id]["results"]
    
    cm_item = next((r for r in first_model_results if r["source"] == "CyberMetric"), None)
    pi_item = next((r for r in first_model_results if r["source"] == "CyberSecEval_PI"), None)
    
    if cm_item:
        story.append(Paragraph("<b>Пример вопроса из CyberMetric (MCQ)</b>", h2_style))
        story.append(Paragraph("<b>Текст вопроса и варианты:</b>", body_style))
        story.append(Paragraph(cm_item["prompt"].replace("\n", "<br/>"), code_style))
        story.append(Paragraph(f"<b>Правильный ответ:</b> {cm_item['reference']}", body_style))
        story.append(Spacer(1, 5))
        
        # Show answers from phi3 and deepseek
        for m_id in ["phi3", "deepseek_r1_1_5b"]:
            if m_id in results_data:
                q_res = next((r for r in results_data[m_id]["results"] if r["id"] == cm_item["id"]), None)
                if q_res:
                    story.append(Paragraph(f"<b>Ответ от модели {m_id}</b> (Оценка: {q_res['accuracy']:.1f}):", body_style))
                    story.append(Paragraph(q_res["response"].replace("\n", "<br/>"), code_style))
                    story.append(Spacer(1, 5))
                    
    if pi_item:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Пример атаки из CyberSecEval Prompt Injection</b>", h2_style))
        story.append(Paragraph("<b>Текст атаки (Промпт):</b>", body_style))
        story.append(Paragraph(pi_item["prompt"].replace("\n", "<br/>"), code_style))
        story.append(Paragraph(f"<b>Секретный ключ (не должен быть раскрыт):</b> {pi_item['reference']}", body_style))
        story.append(Spacer(1, 5))
        
        # Show answers from phi3 and deepseek
        for m_id in ["phi3", "deepseek_r1_1_5b"]:
            if m_id in results_data:
                q_res = next((r for r in results_data[m_id]["results"] if r["id"] == pi_item["id"]), None)
                if q_res:
                    status_text = "БЕЗОПАСЕН (Успех)" if q_res["safety_score"] == 1.0 else "УЯЗВИМ (Утечка ключа!)"
                    story.append(Paragraph(f"<b>Ответ от модели {m_id}</b> ({status_text}):", body_style))
                    
                    resp_text = q_res["response"]
                    if "<think>" in resp_text and "</think>" in resp_text:
                        parts = resp_text.split("</think>")
                        resp_text = f"<i>[Цепочка рассуждений скрыта для краткости]</i><br/><br/>" + parts[1].strip()
                        
                    story.append(Paragraph(resp_text.replace("\n", "<br/>"), code_style))
                    story.append(Spacer(1, 5))
                    
    # Build PDF
    doc.build(story)
    print(f"PDF report successfully generated and saved to {output_pdf_path}")

if __name__ == "__main__":
    results_json = "logs/benchmark/results.json"
    output_pdf = "logs/benchmark/report.pdf"
    charts_png = "logs/benchmark/charts.png"
    
    if os.path.exists(results_json):
        build_pdf_report(results_json, output_pdf, charts_png)
    else:
        print(f"Error: Results file {results_json} not found. Please run run_benchmark.py first.")
        sys.exit(1)
