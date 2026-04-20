"""
medical_report.py
=================
Senior AI Engineer: Generador de Reporte Médico HTML profesional.

Integra predicciones ML + DL en un reporte clínico estructurado con:
  - Dashboard visual (CSS puro, sin dependencias externas)
  - Semáforo de riesgo
  - Cronograma de recuperación
  - Recomendaciones clínicas basadas en IA
  - Metadata de modelo & trazabilidad

Output: reports/medical_report_{player}_{timestamp}.html
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional

from config import Config

# Risk color palette
RISK_COLORS = {
    "Low":      {"bg": "#d1fae5", "border": "#10b981", "text": "#065f46", "badge": "#10b981"},
    "Medium":   {"bg": "#fef9c3", "border": "#f59e0b", "text": "#78350f", "badge": "#f59e0b"},
    "High":     {"bg": "#fee2e2", "border": "#ef4444", "text": "#7f1d1d", "badge": "#ef4444"},
    "Critical": {"bg": "#fce7f3", "border": "#a21caf", "text": "#4a044e", "badge": "#a21caf"},
}

RECOVERY_PROTOCOLS = {
    "Low": [
        ("Days 1-3",  "Rest & RICE protocol (Rest, Ice, Compression, Elevation)"),
        ("Days 3-7",  "Light mobility exercises & physiotherapy assessment"),
        ("Return",    "Gradual return-to-training with monitoring"),
    ],
    "Medium": [
        ("Days 1-5",  "Rest, ice & anti-inflammatory medication as prescribed"),
        ("Days 5-10", "Physiotherapy sessions 3×/week — range of motion work"),
        ("Days 10-21","Progressive load — pool training, cognitive training"),
        ("Return",    "Fitness tests before return to full training"),
    ],
    "High": [
        ("Week 1",    "Complete rest. MRI/ultrasound if not already done"),
        ("Week 2-3",  "Physiotherapy 5×/week. Hydrotherapy, electrostimulation"),
        ("Week 3-5",  "Gym work: non-impact resistance training"),
        ("Week 5+",   "Sport-specific drills, team training integration"),
        ("Return",    "Medical clearance required. Minimum fitness tests + match simulation"),
    ],
    "Critical": [
        ("Week 1-2",  "Surgical consultation if indicated. Complete immobilization"),
        ("Week 2-6",  "Post-op or conservative physiotherapy protocol. Daily sessions"),
        ("Month 2-3", "Progressive weight-bearing, proprioception training"),
        ("Month 3-5", "Gym rehabilitation, strength 80% of contralateral limb target"),
        ("Month 5+",  "Return-to-sport protocol: agility, explosive training"),
        ("Return",    "Multi-disciplinary team clearance required (physician + physio + coach)"),
    ],
}

CLINICAL_RECOMMENDATIONS = {
    "Low":
        "This injury is classified as <strong>low severity</strong>. Standard RICE protocol is recommended. "
        "No advanced imaging is required unless symptoms persist beyond 7 days. "
        "Light physiotherapy may accelerate recovery.",
    "Medium":
        "This injury presents <strong>medium severity</strong>. A structured physiotherapy programme "
        "is strongly recommended. Avoid high-impact loading until mobility is fully restored. "
        "Medical re-evaluation at Day 10 advised.",
    "High":
        "This is a <strong>high-severity injury</strong> requiring immediate specialist referral. "
        "MRI/ultrasound diagnostics are recommended within 48h. The player should not return to "
        "training without written medical clearance. Psychological support may also be beneficial.",
    "Critical":
        "This is a <strong>critical injury</strong>. Immediate orthopaedic or sports medicine consultation "
        "is mandatory. Surgical options must be evaluated. A full multidisciplinary rehabilitation "
        "team should be assembled. Return-to-play timeline should only be communicated after "
        "formal assessment milestones are achieved.",
}


def _probability_bar(label: str, prob: float, color: str) -> str:
    pct = round(prob * 100, 1)
    return f"""
        <div class="prob-row">
            <span class="prob-label">{label}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="prob-value">{pct}%</span>
        </div>"""


def _timeline_items(protocol: list[tuple[str, str]]) -> str:
    items = ""
    for i, (phase, desc) in enumerate(protocol):
        items += f"""
            <div class="timeline-item">
                <div class="timeline-dot {'timeline-dot-last' if i == len(protocol)-1 else ''}"></div>
                <div class="timeline-content">
                    <span class="timeline-phase">{phase}</span>
                    <p class="timeline-desc">{desc}</p>
                </div>
            </div>"""
    return items


def generate_medical_report(
    player_info: dict,
    ml_prediction: dict,
    dl_prediction: Optional[dict] = None,
    ml_metrics: Optional[dict] = None,
    dl_metrics: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Generates a professional HTML medical report.

    Parameters
    ----------
    player_info   : dict with keys: player_name, player_age, player_position,
                    club, league, injury, season
    ml_prediction : dict from InjuryPredictor.predict_player()
    dl_prediction : (optional) dict with 'predicted_days', 'risk_category', 'risk_probabilities'
    ml_metrics    : (optional) dict from InjuryPredictor.train() metrics
    dl_metrics    : (optional) dict from InjuryDLTrainer.train() metrics
    output_dir    : output directory for the HTML file

    Returns
    -------
    str : absolute path to generated HTML file
    """
    config = Config()
    output_dir = output_dir or config.REPORT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    player_slug = player_info.get("player_name", "player").replace(" ", "_").lower()
    filename = f"medical_report_{player_slug}_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    # Core data
    risk = ml_prediction.get("risk_category", "Unknown")
    days = ml_prediction.get("predicted_days", 0)
    probs = ml_prediction.get("risk_probabilities", {})
    colors = RISK_COLORS.get(risk, RISK_COLORS["Medium"])

    # Estimated return date
    return_date = (now + timedelta(days=days)).strftime("%B %d, %Y")

    # DL consensus
    dl_risk = dl_prediction.get("risk_category", "N/A") if dl_prediction else "N/A"
    dl_days = dl_prediction.get("predicted_days", "N/A") if dl_prediction else "N/A"
    consensus_note = ""
    if dl_prediction:
        if dl_risk == risk:
            consensus_note = f'<span class="badge badge-green">✔ DL Consensus: {dl_risk}</span>'
        else:
            consensus_note = (
                f'<span class="badge badge-yellow">⚠ DL Diverges: {dl_risk} '
                f'({dl_days}d) — review advised</span>'
            )

    # Protocol & recommendation
    protocol = RECOVERY_PROTOCOLS.get(risk, RECOVERY_PROTOCOLS["Medium"])
    recommendation = CLINICAL_RECOMMENDATIONS.get(risk, "")

    # Probability bars
    prob_bars = ""
    risk_order = ["Low", "Medium", "High", "Critical"]
    bar_colors = [RISK_COLORS[r]["badge"] for r in risk_order]
    for r_label, b_color in zip(risk_order, bar_colors):
        p = probs.get(r_label, 0)
        prob_bars += _probability_bar(r_label, p, b_color)

    # ML metrics snippet
    ml_metrics_html = ""
    if ml_metrics:
        reg = ml_metrics.get("regression", {})
        clf = ml_metrics.get("classification", {})
        ml_metrics_html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-val">{reg.get('MAE', 'N/A'):.1f}d</span>
                <span class="metric-lbl">ML MAE</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{reg.get('R2', 0):.3f}</span>
                <span class="metric-lbl">ML R²</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{clf.get('accuracy', 0):.3f}</span>
                <span class="metric-lbl">Classifier Acc.</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{clf.get('macro_f1', 0):.3f}</span>
                <span class="metric-lbl">Macro F1</span>
            </div>
        </div>"""

    dl_metrics_html = ""
    if dl_metrics:
        dl_metrics_html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-val">{dl_metrics.get('MAE_days', 0):.1f}d</span>
                <span class="metric-lbl">DL MAE</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{dl_metrics.get('R2_days', 0):.3f}</span>
                <span class="metric-lbl">DL R²</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{dl_metrics.get('accuracy_risk', 0):.3f}</span>
                <span class="metric-lbl">DL Risk Acc.</span>
            </div>
            <div class="metric-card">
                <span class="metric-val">{dl_metrics.get('backend', 'N/A')}</span>
                <span class="metric-lbl">Backend</span>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Injury Medical Report — {player_info.get('player_name','Player')}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #0f172a;
      --surface: #1e293b;
      --surface2: #253045;
      --border: #334155;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #6366f1;
      --accent2: #8b5cf6;
      --risk-bg: {colors['bg']};
      --risk-border: {colors['border']};
      --risk-text: {colors['text']};
    }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}

    /* ─── Header ─── */
    .report-header {{
      background: linear-gradient(135deg, var(--surface) 0%, #1a103a 100%);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 2.5rem;
      margin-bottom: 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1rem;
      position: relative;
      overflow: hidden;
    }}
    .report-header::before {{
      content:'';
      position:absolute;
      top:-60px; right:-60px;
      width:260px; height:260px;
      border-radius:50%;
      background: radial-gradient(circle, rgba(99,102,241,0.25), transparent 70%);
      pointer-events:none;
    }}
    .header-left h1 {{
      font-size:1.6rem; font-weight:800; letter-spacing:-0.5px;
      background: linear-gradient(90deg, #818cf8, #c084fc);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }}
    .header-left .sub {{ color:var(--text-muted); font-size:.85rem; margin-top:4px; }}
    .header-right {{ text-align:right; }}
    .report-id {{ font-size:.75rem; color:var(--text-muted); margin-bottom:4px; }}
    .report-date {{ font-size:.85rem; color:var(--text); font-weight:500; }}
    .generated-by {{
      margin-top:8px;
      display:inline-block;
      font-size:.7rem;
      background: rgba(99,102,241,.15);
      border:1px solid rgba(99,102,241,.3);
      color:#818cf8;
      border-radius:20px;
      padding:3px 10px;
    }}

    /* ─── Risk Banner ─── */
    .risk-banner {{
      background: {colors['bg']};
      border: 2px solid {colors['border']};
      border-radius: 16px;
      padding: 1.5rem 2rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;
      flex-wrap: wrap;
    }}
    .risk-icon {{
      font-size: 3rem;
      line-height:1;
    }}
    .risk-main .risk-label {{
      font-size:.75rem; text-transform:uppercase; letter-spacing:2px;
      color:{colors['text']}; font-weight:600; opacity:.7;
    }}
    .risk-main .risk-value {{
      font-size:2.4rem; font-weight:800; color:{colors['text']}; line-height:1.1;
    }}
    .risk-main .risk-days {{
      font-size:1rem; color:{colors['text']}; opacity:.8;
      margin-top:2px;
    }}
    .risk-return {{
      margin-left:auto;
      text-align:right;
    }}
    .risk-return .ret-label {{
      font-size:.75rem; text-transform:uppercase; letter-spacing:1px;
      color:{colors['text']}; opacity:.7;
    }}
    .risk-return .ret-date {{
      font-size:1.2rem; font-weight:700; color:{colors['text']};
    }}
    .risk-return .consensus {{ margin-top:6px; }}

    /* ─── Card grid ─── */
    .card-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.2rem;
      margin-bottom: 1.2rem;
    }}
    @media(max-width:640px) {{ .card-grid {{ grid-template-columns:1fr; }} }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.4rem;
    }}
    .card-title {{
      font-size:.7rem; text-transform:uppercase; letter-spacing:2px;
      color:var(--text-muted); font-weight:600; margin-bottom:.9rem;
      display:flex; align-items:center; gap:.4rem;
    }}
    .card-title-icon {{ font-size:1rem; }}

    /* Player info table */
    .info-table td {{ padding:.3rem 0; font-size:.9rem; }}
    .info-table td:first-child {{ color:var(--text-muted); width:130px; }}
    .info-table td:last-child {{ font-weight:500; }}

    /* Probability bars */
    .prob-row {{
      display:flex; align-items:center; gap:.7rem; margin-bottom:.6rem; font-size:.85rem;
    }}
    .prob-label {{ width:70px; color:var(--text-muted); flex-shrink:0; }}
    .prob-bar-track {{
      flex:1; height:8px; background:var(--surface2); border-radius:99px; overflow:hidden;
    }}
    .prob-bar-fill {{
      height:100%; border-radius:99px;
      transition: width 0.6s ease;
    }}
    .prob-value {{ width:40px; text-align:right; font-weight:600; font-size:.8rem; }}

    /* ─── Full-width cards ─── */
    .card-full {{ margin-bottom:1.2rem; }}

    /* Timeline */
    .timeline {{ position:relative; padding-left:1.5rem; margin-top:.5rem; }}
    .timeline::before {{
      content:''; position:absolute; left:6px; top:8px;
      width:2px; height:calc(100% - 16px);
      background: linear-gradient(to bottom, #6366f1, transparent);
    }}
    .timeline-item {{
      position:relative; display:flex; gap:1rem; margin-bottom:1rem;
    }}
    .timeline-dot {{
      position:absolute; left:-1.5rem; top:4px;
      width:14px; height:14px; border-radius:50%;
      background: #6366f1; border:2px solid var(--bg); flex-shrink:0;
    }}
    .timeline-dot-last {{ background:#10b981; }}
    .timeline-content {{ flex:1; }}
    .timeline-phase {{
      font-size:.75rem; font-weight:700; color:#818cf8; text-transform:uppercase;
      letter-spacing:1px;
    }}
    .timeline-desc {{ font-size:.85rem; color:var(--text-muted); margin-top:2px; }}

    /* Clinical recommendation box */
    .recommendation-box {{
      background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(139,92,246,.05));
      border: 1px solid rgba(99,102,241,.3);
      border-left: 4px solid #6366f1;
      border-radius: 12px;
      padding: 1.2rem 1.5rem;
      font-size:.9rem; line-height:1.7;
      color: var(--text);
    }}

    /* Metrics grid */
    .metrics-grid {{
      display:grid; grid-template-columns: repeat(4, 1fr); gap:.8rem; margin-top:.5rem;
    }}
    @media(max-width:640px) {{ .metrics-grid {{ grid-template-columns: repeat(2,1fr); }} }}
    .metric-card {{
      background: var(--surface2); border-radius:10px; padding:.9rem;
      text-align:center; border:1px solid var(--border);
    }}
    .metric-val {{ display:block; font-size:1.2rem; font-weight:700; color:#818cf8; }}
    .metric-lbl {{ display:block; font-size:.7rem; color:var(--text-muted);
                   text-transform:uppercase; letter-spacing:1px; margin-top:2px; }}

    /* Badges */
    .badge {{
      display:inline-block; padding:3px 10px; border-radius:99px;
      font-size:.72rem; font-weight:600;
    }}
    .badge-green {{ background:rgba(16,185,129,.2); color:#34d399; border:1px solid #10b981; }}
    .badge-yellow {{ background:rgba(245,158,11,.2); color:#fbbf24; border:1px solid #f59e0b; }}

    /* ─── Footer ─── */
    .report-footer {{
      text-align:center; color:var(--text-muted); font-size:.75rem;
      margin-top:2rem; padding-top:1.5rem; border-top:1px solid var(--border);
    }}
    .disclaimer {{
      background: rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.2);
      border-radius:10px; padding:.9rem 1.2rem; font-size:.78rem;
      color:#fca5a5; line-height:1.6; margin-bottom:1rem;
    }}

    /* Animations */
    @keyframes fadeUp {{
      from {{ opacity:0; transform:translateY(16px); }}
      to   {{ opacity:1; transform:translateY(0);     }}
    }}
    .container > * {{ animation: fadeUp .4s ease both; }}
    .container > *:nth-child(1) {{ animation-delay:.05s; }}
    .container > *:nth-child(2) {{ animation-delay:.1s; }}
    .container > *:nth-child(3) {{ animation-delay:.15s; }}
    .container > *:nth-child(4) {{ animation-delay:.2s; }}
    .container > *:nth-child(5) {{ animation-delay:.25s; }}
  </style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="report-header">
    <div class="header-left">
      <h1>⚕ AI Injury Medical Report</h1>
      <div class="sub">Powered by ML Ensemble + Deep Learning &nbsp;|&nbsp; Sports Medicine AI</div>
    </div>
    <div class="header-right">
      <div class="report-id">Report ID: MED-{timestamp[:8]}-{hash(player_info.get('player_name','x')) % 9999:04d}</div>
      <div class="report-date">{now.strftime("%B %d, %Y · %H:%M")}</div>
      <div class="generated-by">Generated by InjuryAI v2.0</div>
    </div>
  </div>

  <!-- Risk Banner -->
  <div class="risk-banner">
    <div class="risk-icon">{'🔴' if risk=='Critical' else '🟠' if risk=='High' else '🟡' if risk=='Medium' else '🟢'}</div>
    <div class="risk-main">
      <div class="risk-label">Predicted Risk Level</div>
      <div class="risk-value">{risk}</div>
      <div class="risk-days">Estimated recovery: <strong>{days} days</strong></div>
    </div>
    <div class="risk-return">
      <div class="ret-label">Estimated Return Date</div>
      <div class="ret-date">📅 {return_date}</div>
      <div class="consensus">{consensus_note}</div>
    </div>
  </div>

  <!-- Player + Probabilities -->
  <div class="card-grid">
    <div class="card">
      <div class="card-title"><span class="card-title-icon">👤</span> Player Profile</div>
      <table class="info-table">
        <tr><td>Name</td><td>{player_info.get('player_name','—')}</td></tr>
        <tr><td>Age</td><td>{player_info.get('player_age','—')} years</td></tr>
        <tr><td>Position</td><td>{player_info.get('player_position','—')}</td></tr>
        <tr><td>Club</td><td>{player_info.get('club','—')}</td></tr>
        <tr><td>League</td><td>{player_info.get('league','—')}</td></tr>
        <tr><td>Season</td><td>{player_info.get('season','—')}</td></tr>
        <tr><td>Injury</td><td>{player_info.get('injury','—')}</td></tr>
      </table>
    </div>
    <div class="card">
      <div class="card-title"><span class="card-title-icon">📊</span> Risk Probability Distribution</div>
      {prob_bars}
      <div style="margin-top:.8rem;font-size:.75rem;color:var(--text-muted);">
        Probability calibrated classification from ML Ensemble model.
      </div>
    </div>
  </div>

  <!-- Clinical Recommendation -->
  <div class="card card-full">
    <div class="card-title"><span class="card-title-icon">🩺</span> Clinical Recommendation</div>
    <div class="recommendation-box">{recommendation}</div>
  </div>

  <!-- Recovery Timeline -->
  <div class="card card-full">
    <div class="card-title"><span class="card-title-icon">📅</span> Recovery Protocol Timeline</div>
    <div class="timeline">
      {_timeline_items(protocol)}
    </div>
  </div>

  <!-- Model Performance -->
  {'<div class="card card-full"><div class="card-title"><span class="card-title-icon">🤖</span> ML Model Performance (Test Set)</div>' + ml_metrics_html + '</div>' if ml_metrics_html else ''}

  {'<div class="card card-full"><div class="card-title"><span class="card-title-icon">🧠</span> Deep Learning Model Performance</div>' + dl_metrics_html + '</div>' if dl_metrics_html else ''}

  <!-- Disclaimer & Footer -->
  <div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This report is generated by an AI predictive model trained on historical sports injury data.
    It is intended as a <em>decision-support tool only</em> and does not replace the judgement of qualified medical professionals.
    All treatment decisions must be made by licensed physicians and sports medicine specialists.
  </div>
  <div class="report-footer">
    InjuryAI — ML + Deep Learning Sports Medicine Platform &nbsp;·&nbsp; Report ID: MED-{timestamp} &nbsp;·&nbsp;
    This document is confidential and intended solely for the named patient's medical team.
  </div>

</div>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[MedicalReport] ✔ Report saved → {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────
# Entry point — demo report
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo: simulate predictions
    demo_player = {
        "player_name": "Demo Player",
        "player_age": 29,
        "player_position": "Central Midfield",
        "club": "Real Madrid CF",
        "league": "LaLiga Santander",
        "season": "24/25",
        "injury": "Hamstring injury",
    }

    demo_ml_pred = {
        "predicted_days": 21.5,
        "risk_category": "High",
        "risk_probabilities": {"Low": 0.04, "Medium": 0.18, "High": 0.61, "Critical": 0.17},
    }

    demo_dl_pred = {
        "predicted_days": 24.0,
        "risk_category": "High",
        "risk_probabilities": {"Low": 0.02, "Medium": 0.15, "High": 0.65, "Critical": 0.18},
    }

    demo_ml_metrics = {
        "regression": {"MAE": 8.4, "RMSE": 14.2, "R2": 0.612},
        "classification": {"accuracy": 0.7831, "macro_f1": 0.7502},
    }

    path = generate_medical_report(
        player_info=demo_player,
        ml_prediction=demo_ml_pred,
        dl_prediction=demo_dl_pred,
        ml_metrics=demo_ml_metrics,
    )
    print(f"\n[MedicalReport] Open in browser: file:///{path.replace(os.sep, '/')}")
