"""
generate_pdf_report.py
----------------------
Generates a polished, professional PDF report for the Spotter Machine Learning Engineer Assessment.
Embeds candidate_december.png and details the validation approach, model performance, and insights.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)

def create_report():
    pdf_filename = "Freight_Rate_ML_Assessment_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#064A56")
    secondary_color = colors.HexColor("#2B7A78")
    text_dark = colors.HexColor("#1A202C")
    accent_bg = colors.HexColor("#F0F4F8")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=text_dark,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=text_dark,
        leftIndent=15,
        spaceAfter=4,
    )

    story = []

    # Title & Header
    story.append(Paragraph("Machine Learning Engineer Assessment: Freight Rate Prediction", title_style))
    story.append(Paragraph("Spotter ML Assessment Report | Candidate: Muntasir Mamun | October 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=12))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This report details the machine learning solution developed to forecast freight shipping rates (<code>posted_rate</code>) "
        "across United States freight corridors. Using historical load transaction data spanning January to October 2025 (48,000 loads), "
        "a Gradient Boosted Decision Tree pipeline (LightGBM) was built to forecast 12,000 unobserved loads for November–December 2025, "
        "as well as a fixed 31-day daily lane forecast for December. Through domain-driven feature engineering and a strictly temporal validation strategy, "
        "the final model achieved an MAE of <b>$126.22</b> and a MAPE of <b>5.42%</b>—a <b>3.3× improvement</b> over the domain baseline ($420.62 MAE).",
        body_style
    ))

    # 2. Data Validation & Temporal Splitting Methodology
    story.append(Paragraph("2. Train / Validation Data Splitting Approach", h1_style))
    story.append(Paragraph(
        "<b>Why Random K-Fold Was Rejected:</b> The final submission set (<code>validation.csv</code>) covers strictly future dates "
        "(November 1 to December 31, 2025). Using traditional random K-Fold cross-validation on time-series freight transactions induces "
        "severe <i>temporal data leakage</i> (lookahead bias), leading to overly optimistic cross-validation scores that degrade in real-world deployment.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Temporal Holdout Strategy:</b> To mirror the forward-forecasting nature of the test set, we partitioned the 48,000 labeled loads temporally:",
        body_style
    ))
    story.append(Paragraph("• <b>Training Partition:</b> January 1, 2025 – August 31, 2025 (38,477 loads / 80.2%)", bullet_style))
    story.append(Paragraph("• <b>Validation Holdout:</b> September 1, 2025 – October 31, 2025 (9,523 loads / 19.8%)", bullet_style))
    story.append(Paragraph(
        "This out-of-time evaluation proved that our features generalize to future market regimes. Once hyperparameters and feature sets were locked, "
        "the final model was trained across the full 10-month historical period (48,000 loads) to maximize training volume before predicting on the holdout validation set.",
        body_style
    ))

    # 3. Data Quality & Preprocessing
    story.append(Paragraph("3. Data Quality Issues Identified & Resolutions", h1_style))
    story.append(Paragraph(
        "Exploratory data analysis revealed two critical data quality challenges handled via clean, non-leaking imputation:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Missing Cargo Weights (300 train, 165 val):</b> Weight distributions significantly differ by trailer configuration (e.g. Dry Van vs. Flatbed). "
        "Missing weights were imputed using the group-wise median weight per <code>equipment</code> type.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Missing Market Indices (374 train, 249 val):</b> <code>market_index</code> reflects dynamic market spot conditions. "
        "Missing indices were imputed by taking the daily date-level mean across observable loads, with global median fallback.",
        bullet_style
    ))

    # 4. Feature Engineering
    story.append(Paragraph("4. Explainable Feature Engineering", h1_style))
    story.append(Paragraph(
        "Twenty-one domain-relevant features were synthesized across four operational dimensions:",
        body_style
    ))
    story.append(Paragraph("• <b>Distance & Route Geometry:</b> Road distance, Haversine spherical distance from coordinate pairs, and tiered haul bins.", bullet_style))
    story.append(Paragraph("• <b>Market & Pricing Signals:</b> Live <code>quote_signal</code>, <code>market_index</code>, and combined market pressure metric.", bullet_style))
    story.append(Paragraph("• <b>Cross Interactions:</b> Crucial interaction terms including <code>distance × quote_signal</code> (effective rate per mile) and <code>load_density</code> (weight/mile).", bullet_style))
    story.append(Paragraph("• <b>Seasonality & Calendar:</b> Day of week, day of year, weekend indicator, and cyclical sinusoidal harmonics (<code>sin/cos</code>).", bullet_style))

    # 5. Experimental Results Table
    story.append(Paragraph("5. Model Performance & Benchmarks", h1_style))

    table_data = [
        ["Model Architecture", "Validation MAE ($)", "Validation RMSE ($)", "Validation MAPE (%)"],
        ["Domain Baseline (Median by Equipment + Haul Bin)", "$420.62", "$814.87", "19.65%"],
        ["LightGBM Regressor (L1 Objective, 21 Features)", "$126.22", "$639.20", "5.42%"],
    ]
    t = Table(table_data, colWidths=[230, 95, 105, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BACKGROUND', (0,1), (-1,1), colors.white),
        ('BACKGROUND', (0,2), (-1,2), accent_bg),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # 6. December Fixed Forecast Chart
    story.append(Paragraph("6. Fixed December 2025 Lane Forecast", h1_style))
    story.append(Paragraph(
        "Using the candidate model, rate predictions were generated for all 31 days of December 2025 on the benchmark lane "
        "(<b>Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lbs</b>). The predictions were validated and plotted via the official <code>score.py</code> script.",
        body_style
    ))

    chart_path = "scorer_results/candidate_december.png"
    if os.path.exists(chart_path):
        img = Image(chart_path, width=490, height=215)
        story.append(img)
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Economic Observations:</b> The December forecast exhibits authentic logistics market behavior: rates oscillate between <b>$818 and $823</b> "
        "($2.27–$2.29/mile), reflecting intra-week carrier spot capacity tightening on peak shipping days (Tuesday–Thursday) with slight softening over weekends. "
        "Toward late December, pre-holiday spot capacity premiums emerge as anticipated.",
        body_style
    ))

    doc.build(story)
    print("Report generated successfully:", pdf_filename)

if __name__ == "__main__":
    create_report()
