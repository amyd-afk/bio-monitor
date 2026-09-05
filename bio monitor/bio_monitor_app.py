"""
Bio-Monitor Alert System - Week 3 Dashboard
===========================================

A remote patient monitoring dashboard built on the alert logic verified in Week 2.

Monitored conditions:
    1. Acute Hypoxia                    -- SpO2 below safe delivery levels
    2. Severe Tachycardia / Bradycardia -- heart rate outside viable cardiac output

Run locally:  streamlit run bio_monitor_app.py

Intern: [Your Name]
Program: Stimulus Group Services -- Bio-Engineering & Technology Internship
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bio-Monitor Alert System",
    page_icon="+",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Clinical thresholds
#
# Carried over unchanged from the Week 2 alert engine. This block and
# classify_vitals() below are the single source of truth for the whole system --
# the UI never hardcodes a clinical number, so the screen and the alert logic
# cannot drift apart. In a real monitoring product that drift would mean the
# dashboard showing green while the engine considers the patient critical.
# ---------------------------------------------------------------------------

# Oxygen saturation (%)
SPO2_CRITICAL = 90.0   # below this -> Priority 1 on its own
SPO2_COMPOUND = 92.0   # below this -> Priority 1 ONLY when paired with tachycardia
SPO2_WARNING = 95.0    # below this -> Warning

# Heart rate (bpm)
HR_CRITICAL_HIGH = 120.0  # above -> Priority 1 (severe tachycardia)
HR_WARNING_HIGH = 100.0   # above -> Warning; also the compound-rule partner
HR_WARNING_LOW = 60.0     # below -> Warning
HR_CRITICAL_LOW = 50.0    # below -> Priority 1 (severe bradycardia)

# Sensor plausibility -- readings outside these are equipment faults, not patients
VALID_HR_RANGE = (20.0, 300.0)
VALID_SPO2_RANGE = (50.0, 100.0)

# Alert states
NORMAL = "NORMAL"
WARNING = "WARNING"
PRIORITY_1 = "PRIORITY 1"
DATA_ERROR = "DATA ERROR"

# Human-readable labels for the trigger codes the engine returns
TRIGGER_LABELS = {
    "severe_hypoxia": "Severe hypoxia",
    "severe_tachycardia": "Severe tachycardia",
    "severe_bradycardia": "Severe bradycardia",
    "compound_hypoxia_with_tachycardia": "Hypoxia with compensatory tachycardia",
    "low_oxygen_saturation": "Low oxygen saturation",
    "elevated_heart_rate": "Elevated heart rate",
    "low_heart_rate": "Low heart rate",
    "missing_reading": "Sensor fault - missing reading",
    "hr_implausible": "Sensor fault - implausible heart rate",
    "spo2_implausible": "Sensor fault - implausible saturation",
}

STATE_COLORS = {
    NORMAL: "#1a7f4b",
    WARNING: "#b26a00",
    PRIORITY_1: "#c0261f",
    DATA_ERROR: "#5a6470",
}


# ---------------------------------------------------------------------------
# Alert engine  (verbatim from Week 2 -- verified against 18 test cases)
# ---------------------------------------------------------------------------

def _result(status, triggers, hr, spo2):
    """Uniform return shape so callers never branch on structure."""
    return {
        "status": status,
        "triggers": triggers,
        "heart_rate": hr,
        "oxygen_saturation": spo2,
    }


def classify_vitals(heart_rate, oxygen_saturation):
    """
    Screen one set of vital signs against the Week 1 clinical thresholds.

    Returns a dict with keys: status, triggers, heart_rate, oxygen_saturation.
    """
    hr, spo2 = heart_rate, oxygen_saturation

    # --- Stage 1: sensor validation ------------------------------------
    # Runs first so corrupt data can never be scored as clinical fact.
    if pd.isna(hr) or pd.isna(spo2):
        return _result(DATA_ERROR, ["missing_reading"], hr, spo2)

    hr, spo2 = float(hr), float(spo2)

    if not VALID_HR_RANGE[0] <= hr <= VALID_HR_RANGE[1]:
        return _result(DATA_ERROR, ["hr_implausible"], hr, spo2)
    if not VALID_SPO2_RANGE[0] <= spo2 <= VALID_SPO2_RANGE[1]:
        return _result(DATA_ERROR, ["spo2_implausible"], hr, spo2)

    triggers = []

    # --- Stage 2: single-signal Priority 1 ------------------------------
    if spo2 < SPO2_CRITICAL:
        triggers.append("severe_hypoxia")
    if hr > HR_CRITICAL_HIGH:
        triggers.append("severe_tachycardia")
    if hr < HR_CRITICAL_LOW:
        triggers.append("severe_bradycardia")

    # --- Stage 3: compound Priority 1 -----------------------------------
    # Borderline hypoxia is only critical when the heart is already trying
    # to compensate for it.
    if spo2 < SPO2_COMPOUND and hr > HR_WARNING_HIGH:
        triggers.append("compound_hypoxia_with_tachycardia")

    if triggers:
        return _result(PRIORITY_1, triggers, hr, spo2)

    # --- Stage 4: warnings ----------------------------------------------
    if spo2 < SPO2_WARNING:
        triggers.append("low_oxygen_saturation")
    if hr > HR_WARNING_HIGH:
        triggers.append("elevated_heart_rate")
    if hr < HR_WARNING_LOW:
        triggers.append("low_heart_rate")

    if triggers:
        return _result(WARNING, triggers, hr, spo2)

    return _result(NORMAL, [], hr, spo2)


# ---------------------------------------------------------------------------
# Styling
#
# The Priority 1 banner pulses. Motion is used deliberately and only here:
# a static red panel competes with everything else on a busy ward screen,
# whereas movement is detected in peripheral vision. Nothing else on the page
# animates, so the pulse means exactly one thing.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      @keyframes p1pulse {
        0%, 100% { opacity: 1;    box-shadow: 0 0 0 0 rgba(192, 38, 31, 0.55); }
        50%      { opacity: 0.82; box-shadow: 0 0 0 18px rgba(192, 38, 31, 0); }
      }
      .alert-banner {
        border-radius: 10px;
        padding: 1.15rem 1.4rem;
        margin-bottom: 1rem;
        color: #fff;
        font-family: system-ui, -apple-system, sans-serif;
      }
      .alert-p1      { background: #c0261f; animation: p1pulse 1.15s ease-in-out infinite; }
      .alert-warning { background: #b26a00; }
      .alert-normal  { background: #1a7f4b; }
      .alert-error   { background: #5a6470; }
      .alert-title   { font-size: 1.45rem; font-weight: 800; letter-spacing: .02em; }
      .alert-reason  { font-size: 1rem; opacity: .95; margin-top: .35rem; }

      .vital-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        background: rgba(128, 128, 128, 0.10);
        border-left: 6px solid var(--accent);
      }
      .vital-label { font-size: .8rem; text-transform: uppercase;
                     letter-spacing: .06em; opacity: .7; }
      .vital-value { font-size: 2.1rem; font-weight: 700; line-height: 1.15; }
      .vital-range { font-size: .78rem; opacity: .65; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Candidate names for each required field, checked case-insensitively.
# The Kaggle dataset ships under several slightly different schemas, so the
# app detects the columns rather than assuming one spelling.
COLUMN_CANDIDATES = {
    "hr": ["heart rate", "heart_rate", "heartrate", "hr", "pulse"],
    "spo2": ["oxygen saturation", "oxygen_saturation", "spo2",
             "o2 saturation", "oxygen_saturation_%", "saturation"],
    "patient": ["patient id", "patient_id", "patientid", "subject_id", "id"],
    "time": ["timestamp", "time", "datetime", "date", "recorded_at"],
}


def detect_column(df, key):
    """Find the real column name for a logical field, or None."""
    lookup = {c.strip().lower(): c for c in df.columns}
    for candidate in COLUMN_CANDIDATES[key]:
        if candidate in lookup:
            return lookup[candidate]
    # Fall back to a substring match before giving up
    for candidate in COLUMN_CANDIDATES[key]:
        for lower, original in lookup.items():
            if candidate in lower:
                return original
    return None


@st.cache_data(show_spinner="Loading and scoring patient data...")
def load_and_score(file_or_path, cols):
    """
    Read the dataset, clean it, and run every reading through the alert engine.

    Cached so that scoring happens once per file rather than on every
    interaction -- Streamlit re-executes the whole script on each widget change.
    """
    df = pd.read_csv(file_or_path)

    col_hr, col_spo2, col_patient, col_time = cols

    df = df.dropna(subset=[col_hr, col_spo2]).copy()
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    df = df.sort_values([col_patient, col_time]).reset_index(drop=True)

    results = [
        classify_vitals(hr, spo2)
        for hr, spo2 in zip(df[col_hr], df[col_spo2])
    ]
    df["alert_status"] = [r["status"] for r in results]
    df["alert_triggers"] = [r["triggers"] for r in results]

    return df


# ---------------------------------------------------------------------------
# Sidebar - data source and patient selection
# ---------------------------------------------------------------------------

st.sidebar.title("Bio-Monitor")
st.sidebar.caption("Remote patient monitoring - Week 3 prototype")
st.sidebar.divider()

st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader("Vital signs CSV", type=["csv"])

DEFAULT_PATH = "vital_signs.csv"

source = uploaded
if source is None:
    try:
        with open(DEFAULT_PATH, "rb"):
            source = DEFAULT_PATH
        st.sidebar.caption(f"Using bundled `{DEFAULT_PATH}`")
    except OSError:
        st.info(
            "**Upload a vital signs CSV in the sidebar to begin.**\n\n"
            "Use the same Kaggle Human Vital Sign dataset from Weeks 1 and 2."
        )
        st.stop()

# Peek at the header row to detect columns before the full (cached) load
preview = pd.read_csv(source, nrows=5)
if uploaded is not None:
    uploaded.seek(0)

detected = {k: detect_column(preview, k) for k in COLUMN_CANDIDATES}

with st.sidebar.expander("Column mapping", expanded=any(v is None for v in detected.values())):
    st.caption("Detected automatically. Override if the dataset uses other names.")
    options = list(preview.columns)

    def picker(label, key):
        found = detected[key]
        index = options.index(found) if found in options else 0
        return st.selectbox(label, options, index=index, key=f"col_{key}")

    col_hr = picker("Heart rate", "hr")
    col_spo2 = picker("Oxygen saturation", "spo2")
    col_patient = picker("Patient ID", "patient")
    col_time = picker("Timestamp", "time")

df = load_and_score(source, (col_hr, col_spo2, col_patient, col_time))

st.sidebar.divider()
st.sidebar.subheader("Patient")

patient_ids = sorted(df[col_patient].unique())

# Surface the patients in trouble first -- on a real ward nobody scrolls a
# dropdown of 200 stable patients looking for the one who is deteriorating.
p1_counts = (
    df[df["alert_status"] == PRIORITY_1][col_patient]
    .value_counts()
    .to_dict()
)


def patient_label(pid):
    n = p1_counts.get(pid, 0)
    return f"{pid}  ({n} alerts)" if n else f"{pid}"


if st.sidebar.checkbox("Show only patients with alerts", value=bool(p1_counts)):
    shown = [p for p in patient_ids if p in p1_counts] or patient_ids
else:
    shown = patient_ids

selected = st.sidebar.selectbox(
    "Patient ID",
    shown,
    format_func=patient_label,
)

# ---------------------------------------------------------------------------
# Sidebar - live playback controls
#
# The dataset is historical, so "real time" is simulated by replaying one
# patient's timeline reading by reading. This is what the monitor would look
# like receiving a live sensor feed.
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Live monitor")

patient_df = df[df[col_patient] == selected].reset_index(drop=True)
n_readings = len(patient_df)

# Reset playback whenever the clinician switches patients
if st.session_state.get("current_patient") != selected:
    st.session_state.current_patient = selected
    st.session_state.idx = n_readings - 1
    st.session_state.playing = False

st.session_state.setdefault("idx", n_readings - 1)
st.session_state.setdefault("playing", False)

mode = st.sidebar.radio(
    "Mode",
    ["Full history", "Live replay"],
    help="Live replay streams the patient's readings one at a time.",
)

if mode == "Live replay":
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Play" if not st.session_state.playing else "Pause",
                 use_container_width=True):
        st.session_state.playing = not st.session_state.playing
        st.rerun()
    if c2.button("Restart", use_container_width=True):
        st.session_state.idx = 0
        st.session_state.playing = False
        st.rerun()

    st.session_state.idx = st.sidebar.slider(
        "Reading", 0, max(n_readings - 1, 0), st.session_state.idx
    )
    speed = st.sidebar.select_slider(
        "Speed", options=[0.5, 1, 2, 4, 8], value=2,
        format_func=lambda s: f"{s}x",
    )
    cursor = st.session_state.idx
else:
    st.session_state.playing = False
    cursor = n_readings - 1

st.sidebar.divider()
with st.sidebar.expander("Active thresholds"):
    st.markdown(
        f"""
        **Priority 1**
        - SpO2 < {SPO2_CRITICAL:.0f}%
        - SpO2 < {SPO2_COMPOUND:.0f}% **and** HR > {HR_WARNING_HIGH:.0f} bpm
        - HR > {HR_CRITICAL_HIGH:.0f} bpm
        - HR < {HR_CRITICAL_LOW:.0f} bpm

        **Warning**
        - SpO2 < {SPO2_WARNING:.0f}%
        - HR > {HR_WARNING_HIGH:.0f} or < {HR_WARNING_LOW:.0f} bpm
        """
    )


# ---------------------------------------------------------------------------
# Main panel - header and ward summary
# ---------------------------------------------------------------------------

st.title("Bio-Monitor Alert System")

ward = df["alert_status"].value_counts()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Patients monitored", f"{df[col_patient].nunique():,}")
m2.metric("Readings scored", f"{len(df):,}")
m3.metric("Priority 1 events", f"{int(ward.get(PRIORITY_1, 0)):,}")
m4.metric("Patients in alert", f"{len(p1_counts):,}")

st.divider()


# ---------------------------------------------------------------------------
# Current reading and alert banner
# ---------------------------------------------------------------------------

if n_readings == 0:
    st.warning("No readings for this patient.")
    st.stop()

row = patient_df.iloc[cursor]
hr_now = row[col_hr]
spo2_now = row[col_spo2]
result = classify_vitals(hr_now, spo2_now)
status = result["status"]

reasons = ", ".join(TRIGGER_LABELS.get(t, t) for t in result["triggers"])

banner_class = {
    PRIORITY_1: "alert-p1",
    WARNING: "alert-warning",
    NORMAL: "alert-normal",
    DATA_ERROR: "alert-error",
}[status]

banner_title = {
    PRIORITY_1: "PRIORITY 1 ALERT - IMMEDIATE RESPONSE REQUIRED",
    WARNING: "WARNING - MONITOR CLOSELY",
    NORMAL: "STABLE - ALL VITALS WITHIN RANGE",
    DATA_ERROR: "SENSOR FAULT - PATIENT STATUS UNKNOWN",
}[status]

subtitle = reasons if reasons else "No thresholds breached."
if status == DATA_ERROR:
    subtitle += "  This is not a clinical reading - verify the equipment."

st.markdown(
    f"""
    <div class="alert-banner {banner_class}">
      <div class="alert-title">{banner_title}</div>
      <div class="alert-reason">Patient {selected} &nbsp;-&nbsp; {subtitle}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Current vitals -------------------------------------------------------


def vital_card(label, value, unit, accent, ref):
    display = "-" if pd.isna(value) else f"{value:.0f}"
    return f"""
    <div class="vital-card" style="--accent:{accent};">
      <div class="vital-label">{label}</div>
      <div class="vital-value">{display}<span style="font-size:1rem;"> {unit}</span></div>
      <div class="vital-range">{ref}</div>
    </div>
    """


def signal_color(kind, value):
    """Colour a single vital by its own thresholds, independent of overall status."""
    if pd.isna(value):
        return STATE_COLORS[DATA_ERROR]
    if kind == "spo2":
        if value < SPO2_CRITICAL:
            return STATE_COLORS[PRIORITY_1]
        if value < SPO2_WARNING:
            return STATE_COLORS[WARNING]
    else:
        if value > HR_CRITICAL_HIGH or value < HR_CRITICAL_LOW:
            return STATE_COLORS[PRIORITY_1]
        if value > HR_WARNING_HIGH or value < HR_WARNING_LOW:
            return STATE_COLORS[WARNING]
    return STATE_COLORS[NORMAL]


v1, v2, v3 = st.columns(3)
v1.markdown(
    vital_card("Oxygen saturation", spo2_now, "%",
               signal_color("spo2", spo2_now), f"Normal {SPO2_WARNING:.0f}-100%"),
    unsafe_allow_html=True,
)
v2.markdown(
    vital_card("Heart rate", hr_now, "bpm",
               signal_color("hr", hr_now),
               f"Normal {HR_WARNING_LOW:.0f}-{HR_WARNING_HIGH:.0f} bpm"),
    unsafe_allow_html=True,
)
timestamp = row[col_time]
stamp = "-" if pd.isna(timestamp) else str(timestamp)
v3.markdown(
    f"""
    <div class="vital-card" style="--accent:{STATE_COLORS[NORMAL]};">
      <div class="vital-label">Reading</div>
      <div class="vital-value">{cursor + 1}<span style="font-size:1rem;"> / {n_readings}</span></div>
      <div class="vital-range">{stamp}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------

visible = patient_df.iloc[: cursor + 1]
alerts = visible[visible["alert_status"] == PRIORITY_1]


def trend_chart(y_col, title, unit, line_color, bands, y_range):
    """
    One vital over time, with clinical threshold bands shaded behind the trace
    and Priority 1 readings marked.

    Shaded bands rather than plain lines: a clinician should be able to see
    that a value is in dangerous territory without reading the axis.
    """
    fig = go.Figure()

    for lo, hi, color in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.13,
                      line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=visible.index, y=visible[y_col],
        mode="lines", name=title,
        line=dict(color=line_color, width=2),
        hovertemplate=f"Reading %{{x}}<br>%{{y:.1f}} {unit}<extra></extra>",
    ))

    if not alerts.empty:
        fig.add_trace(go.Scatter(
            x=alerts.index, y=alerts[y_col],
            mode="markers", name="Priority 1",
            marker=dict(color=STATE_COLORS[PRIORITY_1], size=9,
                        line=dict(color="white", width=1)),
            hovertemplate=f"ALERT<br>%{{y:.1f}} {unit}<extra></extra>",
        ))

    # Marker for where the live cursor currently sits
    fig.add_vline(x=cursor, line_width=1.5, line_dash="dot", line_color="#888")

    fig.update_layout(
        title=title,
        height=290,
        margin=dict(l=10, r=10, t=42, b=10),
        showlegend=False,
        yaxis=dict(title=unit, range=y_range),
        xaxis=dict(title="Reading number"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


st.plotly_chart(
    trend_chart(
        col_spo2, "Oxygen saturation", "%", "#2c7fb8",
        bands=[
            (VALID_SPO2_RANGE[0], SPO2_CRITICAL, STATE_COLORS[PRIORITY_1]),
            (SPO2_CRITICAL, SPO2_WARNING, STATE_COLORS[WARNING]),
            (SPO2_WARNING, 100, STATE_COLORS[NORMAL]),
        ],
        y_range=[80, 101],
    ),
    use_container_width=True,
)

st.plotly_chart(
    trend_chart(
        col_hr, "Heart rate", "bpm", "#2e8b57",
        bands=[
            (HR_CRITICAL_HIGH, 200, STATE_COLORS[PRIORITY_1]),
            (HR_WARNING_HIGH, HR_CRITICAL_HIGH, STATE_COLORS[WARNING]),
            (HR_WARNING_LOW, HR_WARNING_HIGH, STATE_COLORS[NORMAL]),
            (HR_CRITICAL_LOW, HR_WARNING_LOW, STATE_COLORS[WARNING]),
            (0, HR_CRITICAL_LOW, STATE_COLORS[PRIORITY_1]),
        ],
        y_range=[30, 160],
    ),
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Alert log
# ---------------------------------------------------------------------------

st.divider()
st.subheader(f"Alert log - Patient {selected}")

log = visible[visible["alert_status"].isin([PRIORITY_1, WARNING, DATA_ERROR])].copy()

if log.empty:
    st.success("No alerts recorded for this patient in the readings shown.")
else:
    log["Reason"] = log["alert_triggers"].apply(
        lambda ts: ", ".join(TRIGGER_LABELS.get(t, t) for t in ts)
    )
    display = log[[col_time, col_hr, col_spo2, "alert_status", "Reason"]].rename(
        columns={
            col_time: "Time",
            col_hr: "HR (bpm)",
            col_spo2: "SpO2 (%)",
            "alert_status": "Status",
        }
    )
    st.dataframe(
        display.sort_index(ascending=False),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    counts = log["alert_status"].value_counts()
    st.caption(
        f"{int(counts.get(PRIORITY_1, 0))} Priority 1 - "
        f"{int(counts.get(WARNING, 0))} Warning - "
        f"{int(counts.get(DATA_ERROR, 0))} Sensor fault"
    )

st.caption(
    "Prototype for educational use. Thresholds assume a general adult population "
    "and are not valid for COPD or pediatric patients without per-patient profiles."
)


# ---------------------------------------------------------------------------
# Playback tick
#
# Advances one reading then reruns the script. Placed last so the full frame
# renders before the delay -- otherwise the page would appear to stall.
# ---------------------------------------------------------------------------

if st.session_state.playing and mode == "Live replay":
    if cursor < n_readings - 1:
        time.sleep(1.0 / speed)
        st.session_state.idx = cursor + 1
        st.rerun()
    else:
        st.session_state.playing = False
