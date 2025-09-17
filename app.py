import os
import time
import uuid
import base64
import logging
import numpy as np
import pandas as pd
from openai import OpenAI
from datetime import datetime
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ================== CONFIG ==================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

LOCAL_FILE_PATH = "sensor_data_log.csv"
ATTENTION_LOG_DIR = "attention_logs"
MEETING_SUMMARY_DIR = "summary_reports"

os.makedirs(ATTENTION_LOG_DIR, exist_ok=True)
os.makedirs(MEETING_SUMMARY_DIR, exist_ok=True)

# ================== DATA ==================

users_data = {}

SCAN_CHECKPOINTS = [15, 35, 55, 75, 95, 115] 
fired_checkpoints = set()

LABEL_TO_ATTENTION = {
    0: "attentive", 1: "attentive", 2: "attentive", 3: "attentive",
    4: "attentive", 5: "attentive", 6: "attentive",
    7: "distracted", 8: "distracted", 9: "distracted", 10: "distracted", 11: "distracted"
}

meeting_state = {
    "meeting_active": False,
    "start_time": None,
    "expected_users": set(),   
    "ended_users": set(),      
    "snapshot": {}             
}

# ================== HELPERS ==================
def triangular_score(x, m, r):
    """
    Triangular score in [0,1]. Return None when x is missing.
    """
    if x is None:
        return None
    return max(0.0, 1.0 - abs(x - m) / r)

def compute_meeting_productivity(attentiveness, sensors,
                                 w_A=0.8, w_E=0.2,
                                 env_weights=None):
    """
    Compute Meeting Productivity Score (MPS).
    - attentiveness: list of percentages [0–100]
    - sensors: dict with keys 'light', 'temp', 'humidity', 'co2' (raw readings)
    Returns:
      {'A': float, 'E': float, 'MPS': float, 'env_scores': {...}}
      where A, E, MPS are percentages (0-100).
    """
    if env_weights is None:
        env_weights = {'light': 0.15, 'temp': 0.25, 'humidity': 0.15, 'co2': 0.45}

    ideals = {
        'light':   (400, 200),   # lux
        'temp':    (22, 2),      # °C
        'humidity':(45, 15),     # %
        'co2':     (600, 400)    # ppm
    }

    # Attentiveness normalized to [0,1]
    att_norm = (np.array(attentiveness) / 100.0) if len(attentiveness) > 0 else np.array([0.0])
    A_frac = float(np.mean(att_norm))

    # Per-sensor triangular scores (0..1), ignore missing sensors
    s_env = {}
    used_weights = {}
    for k, (m, r) in ideals.items():
        score = triangular_score(sensors.get(k), m, r)
        if score is not None:
            s_env[k] = score
            used_weights[k] = env_weights.get(k, 0.0)

    # If at least one sensor present, compute weighted average and normalize by used weight sum.
    if used_weights:
        weight_sum = sum(used_weights.values())
        # avoid division by zero (shouldn't happen if used_weights non-empty)
        E_frac = sum((used_weights[k] * s_env[k] for k in used_weights)) / weight_sum if weight_sum > 0 else 0.5
    else:
        # no sensor data → neutral environment
        E_frac = 0.5

    # Final MPS fraction
    MPS_frac = w_A * A_frac + w_E * E_frac

    # Convert fractions to percentages
    return {
        'A': round(A_frac * 100.0, 2),
        'E': round(E_frac * 100.0, 2),
        'MPS': round(MPS_frac * 100.0, 2),
        'env_scores': {k: round(v * 100.0, 1) for k, v in s_env.items()}
    }

def update_user_attention(user_id, attention_status):
    """
    Update user stats by new status.
    """
    if not isinstance(user_id, str):
        logging.error(f"UserID must be a string! Got type: {type(user_id)} value: {user_id}")
        return
    if user_id not in users_data:
        logging.info(f"🆕 New user detected: {user_id}. Initializing data.")
        users_data[user_id] = {"last_status": attention_status, "attentive_count": 0, "total_scans": 0, "meeting_active": True}

    user = users_data[user_id]
    user["total_scans"] += 1
    
    if attention_status == "attentive":
        user["attentive_count"] += 1
        logging.info(f"✅ User {user_id}: Status 'attentive'. attentive_count={user['attentive_count']}")
    else:
        logging.info(f"⚠️ User {user_id}: Status '{attention_status}'")
        
    user["last_status"] = attention_status
    
    logging.info(f"📊 Current: attentive_count={user['attentive_count']}, total_scans={user['total_scans']}")

    check_global_checkpoints()
    
def log_request_response(route, request_data, response_data):
    logging.info(f"[{route}] Request: {request_data}")
    logging.info(f"[{route}] Response: {response_data}")

def generate_ambient_plot(df, user_id):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    metrics = {
        "Temperature (°C)": pd.to_numeric(df["Temperature"], errors='coerce').mean(),
        "Humidity (%)": pd.to_numeric(df["Humidity"], errors='coerce').mean(),
        "Light": pd.to_numeric(df["Light Intensity"], errors='coerce').mean(),
        "CO₂ (ppm)": pd.to_numeric(df["Co2 Concentration"], errors='coerce').mean()
    }
    
    ax.bar(metrics.keys(), metrics.values(), color=["#F45D5D","#449BF2","#66F866","#E7A461"])
    ax.set_title(f"Ambient Conditions for User {user_id}", fontsize=12, weight="bold")
    plt.xticks(rotation=30, ha="right")
    
    # Unique file path
    filename = f"{MEETING_SUMMARY_DIR}/ambient_{user_id}_{uuid.uuid4().hex[:6]}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    
    with open(filename, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    return {"file_path": filename, "base64": image_base64}

def build_suggestion(attentive_percent, avg_temp, avg_humidity, avg_light, avg_co2, door_status, motion_status, llm_tip):
    base_suggestion = (
        f"📊 Ambient Report:\n"
        f"- Temperature: {avg_temp:.1f}°C\n"
        f"- Humidity: {avg_humidity:.1f}%\n"
        f"- Light: {avg_light:.1f}\n"
        f"- CO₂: {avg_co2:.0f} ppm\n\n"
        f"👀 Attention:\n"
        f"Your attentive percent was {attentive_percent:.1f}%.\n\n"
        f"💡{llm_tip}"
    )
    
    return base_suggestion




def check_global_checkpoints():
    if not users_data:
        return
    min_scans = min(u["total_scans"] for u in users_data.values())
    for cp in SCAN_CHECKPOINTS:
        if min_scans >= cp and cp not in fired_checkpoints:
            logging.info(f"📌 Global checkpoint reached: {cp} scans (all users ≥ {cp})")
            
            # Generate checkpoint report (popup scan)
            checkpoint_data = _collect_owner_summary_data()
            checkpoint_file, plots = _save_checkpoint_files(checkpoint_data, cp)
            
            popup = {
                "type": "checkpoint_popup",
                "checkpoint": cp,
                "overall_attentive_percent": checkpoint_data["overall_attentive_percent"],
                "productive": checkpoint_data["productive"],
                "summary_file": checkpoint_file,
                "plots": plots,
                "users": checkpoint_data["user_summaries"]
            }
            logging.info(f"✅ Checkpoint popup generated: {popup}")
            
            fired_checkpoints.add(cp)

def _save_checkpoint_files(summary_data, checkpoint_number):
    """
    Save checkpoint CSV and plots (similar to owner summary but for checkpoints).
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # === CSV checkpoint summary ===
    checkpoint_file = os.path.join(MEETING_SUMMARY_DIR, f"checkpoint_{checkpoint_number}_summary_{timestamp}.csv")
    summary_df = pd.DataFrame(summary_data["user_summaries"])
    summary_df["overall_attentive_percent"] = summary_data["overall_attentive_percent"]
    summary_df.to_csv(checkpoint_file, index=False)
    logging.info(f"✅ Saved checkpoint {checkpoint_number} CSV: {checkpoint_file}")
    
    user_ids = [u["user_id"] for u in summary_data["user_summaries"]]
    attentive_percents = [float(u["attentive_percent"]) if u["attentive_percent"] is not None else 0.0
                        for u in summary_data["user_summaries"]]

    # === Bar chart for checkpoint ===
    bar_path = os.path.join(MEETING_SUMMARY_DIR, f"attention_percent_popup_scan_{checkpoint_number}.png")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors 
    bars = plt.bar(user_ids, attentive_percents, color=colors[:len(user_ids)])
    plt.ylabel("Attentive Percent")
    plt.ylim(0, 100)
    plt.title(f"Checkpoint {checkpoint_number}: Per-user Attentive % (Overall: {summary_data['overall_attentive_percent']:.1f}%)")
    plt.xticks(rotation=30, ha='right')

    for bar, percent in zip(bars, attentive_percents):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{percent:.1f}%",
                 ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()
    logging.info(f"✅ Saved checkpoint bar chart: {bar_path}")

    # === Pie chart for checkpoint ===
    pie_path = os.path.join(MEETING_SUMMARY_DIR, f"attention_split_popup_scan_{checkpoint_number}.png")
    non_attentive = summary_data["total_scans"] - summary_data["total_attentive"]
    plt.figure(figsize=(5, 5))

    if summary_data["total_scans"] > 0:
        plt.pie(
            [summary_data["total_attentive"], non_attentive],
            labels=['Attentive', 'Not attentive'],
            autopct=lambda p: f'{p:.1f}%\n({int(round(p*summary_data["total_scans"]/100))})',
            colors=["#2E8B57", "#E07B39"],
            startangle=90
        )
    else:
        plt.pie([1], labels=['No data'], colors=['gray'])

    plt.title(f"Checkpoint {checkpoint_number}: Overall Attention Split")
    plt.tight_layout()
    plt.savefig(pie_path, dpi=150)
    plt.close()
    logging.info(f"✅ Saved checkpoint pie chart: {pie_path}")

    return checkpoint_file, [bar_path, pie_path]

def _collect_owner_summary_data():
    """
    Build owner-level summary from current users_data.
    """
    total_attentive = 0
    total_scans = 0
    user_summaries = []

    for user_id, u in users_data.items():
        scans = u.get("total_scans", 0)
        att = u.get("attentive_count", 0)
        percent = (att / scans * 100) if scans > 0 else 0.0
        user_summaries.append({
            "user_id": user_id,
            "attentive_count": att,
            "total_scans": scans,
            "attentive_percent": round(float(percent), 1),
        })
        total_attentive += att
        total_scans += scans

    overall_percent = round((total_attentive / total_scans * 100), 1) if total_scans > 0 else 0.0

    # Load environment window safely
    try:
        df = pd.read_csv(LOCAL_FILE_PATH).tail(300)
        avg_temp = pd.to_numeric(df["Temperature"], errors="coerce").mean()
        avg_humidity = pd.to_numeric(df["Humidity"], errors="coerce").mean()
        avg_light = pd.to_numeric(df["Light Intensity"], errors="coerce").mean()
        avg_co2 = pd.to_numeric(df["Co2 Concentration"], errors="coerce").mean()
    except Exception as e:
        logging.warning(f"Owner summary: sensor file not available or malformed: {e}")
        avg_temp = avg_humidity = avg_light = avg_co2 = None

    productivity_score = compute_meeting_productivity(
        [u["attentive_percent"] for u in user_summaries],
        {"light": avg_light, "temp": avg_temp, "humidity": avg_humidity, "co2": avg_co2}
    )

    return {
        "overall_attentive_percent": overall_percent,
        "productive": productivity_score["MPS"] >= 60,
        "productivity": productivity_score,
        "user_summaries": user_summaries,
        "total_attentive": total_attentive,
        "total_scans": total_scans
    }

def _save_owner_summary_files(summary_data):
    """
    Save final meeting summary CSV and plots.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # === CSV summary ===
    summary_file = os.path.join(MEETING_SUMMARY_DIR, f"meeting_summary_{timestamp}.csv")
    summary_df = pd.DataFrame(summary_data["user_summaries"])
    summary_df["overall_attentive_percent"] = summary_data["overall_attentive_percent"]
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"✅ Saved final meeting summary CSV: {summary_file}")

    user_ids = [u["user_id"] for u in summary_data["user_summaries"]]
    attentive_percents = [float(u["attentive_percent"]) if u["attentive_percent"] is not None else 0.0
                        for u in summary_data["user_summaries"]]

    # === Bar chart ===
    bar_path = os.path.join(MEETING_SUMMARY_DIR, f"attentive_percent_{timestamp}.png")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors 
    bars = plt.bar(user_ids, attentive_percents, color=colors[:len(user_ids)])
    plt.ylabel("Attentive Percent")
    plt.ylim(0, 100)
    plt.title(f"Per-user Attentive % (Overall: {summary_data['overall_attentive_percent']:.1f}%)")
    plt.xticks(rotation=30, ha='right')

    for bar, percent in zip(bars, attentive_percents):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{percent:.1f}%",
                 ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()
    logging.info(f"✅ Saved final bar chart: {bar_path}")

    # === Pie chart ===
    pie_path = os.path.join(MEETING_SUMMARY_DIR, f"attention_split_{timestamp}.png")
    non_attentive = summary_data["total_scans"] - summary_data["total_attentive"]
    plt.figure(figsize=(5, 5))

    if summary_data["total_scans"] > 0:
        plt.pie(
            [summary_data["total_attentive"], non_attentive],
            labels=['Attentive', 'Not attentive'],
            autopct=lambda p: f'{p:.1f}%\n({int(round(p*summary_data["total_scans"]/100))})',
            colors=["#2E8B57", "#E07B39"],
            startangle=90
        )
    else:
        plt.pie([1], labels=['No data'], colors=['gray'])

    plt.title("Final Overall Attention Split")
    plt.tight_layout()
    plt.savefig(pie_path, dpi=150)
    plt.close()
    logging.info(f"✅ Saved final pie chart: {pie_path}")

    # === Combined chart ===    
    combined_chart_path = os.path.join(MEETING_SUMMARY_DIR, f"combined_summary_{timestamp}.png")
    plot_combined_summary(
        summary_data["user_summaries"],
        summary_data["overall_attentive_percent"],
        summary_data["productivity"]["MPS"],   
        combined_chart_path
    )

    logging.info(f"✅ Saved final combined summary: {combined_chart_path}")
    logging.info(f"📊 Final Meeting Productivity Score (MPS): {summary_data['productivity']['MPS']:.1f}%")

    return summary_file, [bar_path, pie_path, combined_chart_path]

def plot_combined_summary(users_data, overall_percent, mps_value, output_path):
    user_names = [u['user_id'] for u in users_data]
    percents = [u['attentive_percent'] for u in users_data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # === Subplot 1: per-user horizontal bars ===
    ax1 = axes[0]
    colors = [ "#4E79A7" if p >= 70 else ( "#F28E2B" if p >= 50 else  "#76B7B2") for p in percents]
    bars = ax1.barh(user_names, percents, color=colors)
    ax1.set_xlim(0, 100)
    ax1.set_title("Per-user Involvement", fontsize=12, fontweight='bold')

    for bar, p in zip(bars, percents):
        status = "Highly Involved" if p >= 70 else ("Moderate" if p >= 50 else "Lacking Behind")
        ax1.text(p + 2, bar.get_y() + bar.get_height()/2, status, va='center', fontsize=10, fontweight='bold')

    # === Subplot 2: pie chart ===
    ax2 = axes[1]
    ax2.pie(
        [overall_percent, 100 - overall_percent],
        labels=['Involved', 'Not Involved'],
        colors= ["#2E8B57", "#E07B39"],
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title("Overall Attention", fontsize=12, fontweight='bold')

    # === Subplot 3: title / suggestion text ===
    ax3 = axes[2]
    ax3.axis('off')
    main_msg = "Class is heavily involved" if overall_percent >= 70 else "Attention Needs Improvement"
    sub_msg = "Keep Going!" if overall_percent >= 70 else "Try Engagement Strategies"

    ax3.text(0.5, 0.7, main_msg, fontsize=16, fontweight='bold', ha='center')
    ax3.text(0.5, 0.4, sub_msg, fontsize=14, ha='center')

    # === Add MPS at the very top of the figure ===
    fig.suptitle(f"Meeting Productivity Score (MPS): {mps_value:.1f}%", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved combined summary chart: {output_path}")

def _collect_owner_summary_from_snapshot(snapshot_dict):
    user_summaries = []
    total_attentive = 0
    total_scans = 0

    for user_id, rec in snapshot_dict.items():
        p = float(rec.get("attentive_percent", 0.0))
        c_att = int(rec.get("attentive_count", 0))
        c_tot = int(rec.get("total_scans", 0))
        user_summaries.append({
            "user_id": user_id,
            "attentive_count": c_att,
            "total_scans": c_tot,
            "attentive_percent": p
        })
        total_attentive += c_att
        total_scans += c_tot

    overall_percent = (total_attentive / total_scans * 100) if total_scans > 0 else 0.0
    overall_percent = round(overall_percent, 1)

    # Load latest environment window
    df = pd.read_csv(LOCAL_FILE_PATH).tail(300)
    avg_temp = pd.to_numeric(df["Temperature"], errors="coerce").mean()
    avg_humidity = pd.to_numeric(df["Humidity"], errors="coerce").mean()
    avg_light = pd.to_numeric(df["Light Intensity"], errors="coerce").mean()
    avg_co2 = pd.to_numeric(df["Co2 Concentration"], errors="coerce").mean()

    productivity_score = compute_meeting_productivity(
        [u["attentive_percent"] for u in user_summaries],
        {
            "light": avg_light,
            "temp": avg_temp,
            "humidity": avg_humidity,
            "co2": avg_co2
        }
    )

    return {
        "overall_attentive_percent": overall_percent,
        "productive": bool(productivity_score["MPS"] >= 60),  
        "user_summaries": user_summaries,
        "total_attentive": total_attentive,
        "total_scans": total_scans,
        "productivity": {
            "A": float(productivity_score["A"]),
            "E": float(productivity_score["E"]),
            "MPS": float(productivity_score["MPS"]),
            "env_scores": {k: float(v) for k, v in productivity_score["env_scores"].items()}
        }
    }

# ================== ROUTES ==================
    
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    user_id = payload.get("UserID")
    probabilities = payload.get("probabilities")

    logging.info(f"Payload size: {len(str(payload))} bytes")
    if not isinstance(user_id, str):
        logging.error(f"/predict: UserID not a string! Got type: {type(user_id)}, value: {user_id}")
        return jsonify(error="UserID must be a string!"), 400

    if not (isinstance(probabilities, list) and all(isinstance(p, (int, float)) for p in probabilities)):
        logging.error("/predict: probabilities must be a list of numbers! Got: %s", type(probabilities))
        return jsonify(error="probabilities must be a list of numbers!"), 400

    if len(probabilities) != 12:
        logging.error(f"/predict: probabilities must be of length 12 (got {len(probabilities)})")
        return jsonify(error="probabilities must be a list of length 12!"), 400

    try:
        probs = np.array(probabilities, dtype=np.float32)
        label = int(np.argmax(probs))
        attention_status = LABEL_TO_ATTENTION.get(label, "unknown")
        update_user_attention(user_id, attention_status)
        logging.info("📊 Current users_data map:")
        for uid, stats in users_data.items():
            logging.info(f" - UserID: {uid} → attentive_count={stats['attentive_count']}, total_scans={stats['total_scans']}, last_status={stats['last_status']}")
    except Exception as e:
        logging.exception(f"❌ Exception occurred in /predict: {e}")
        return jsonify(error=str(e)), 500

    return jsonify({
        "probabilities": probs.tolist(),
        "predicted_label": label,
        "attention_status": attention_status,
        "user_data": users_data[user_id]
    })

@app.route("/start_meeting", methods=["POST"])
def start_meeting():
    data = request.get_json(force=True)
    user_id = data["user_id"]

    # reset for new session
    users_data[user_id] = {
        "last_status": "unknown",
        "attentive_count": 0,
        "total_scans": 0,
        "meeting_active": True
    }

    # meeting coordinator
    meeting_state["meeting_active"] = True
    meeting_state["expected_users"].add(user_id)
    meeting_state["ended_users"].discard(user_id)
    if meeting_state["start_time"] is None:
        meeting_state["start_time"] = datetime.now()

    logging.info(f"🚀 Meeting started for {user_id}. Expected users: {sorted(meeting_state['expected_users'])}")
    
    return jsonify({"status": "started", "user_id": user_id})
    
@app.route("/attention", methods=["POST"])
def log_attention():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    status = data.get("attention_status")
    
    if not isinstance(user_id, str) or not isinstance(status, str):
        return jsonify({"error": "Invalid user_id or attention_status"}), 400

    update_user_attention(user_id, status)
    resp = {"message": f"✅ Logged new status: {status}", "data": users_data[user_id]}
    
    return jsonify(resp)

@app.route("/attention_poke", methods=["POST"])
def attention_poke():
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id")
        
        if not user_id or not isinstance(user_id, str):
            return jsonify({"error": "Missing or invalid user_id"}), 400

        user = users_data.get(user_id)
        if not user:
            return jsonify({"error": "No such user yet"}), 404

        last_status = user["last_status"]
        update_user_attention(user_id, last_status)

        resp = {"message": f"Poked → counted last_status '{last_status}'", "user_data": users_data[user_id]}

        return jsonify(resp)

    except Exception as e:
        logging.error(f"❌ Error in /attention_poke: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/end_meeting", methods=["POST"])
def end_meeting():
    try:
        data = request.get_json(force=True)
        user_id = data["user_id"]
        logging.info(f"End Meeting Request Payload Size : {len(str(data))} bytes")

        user = users_data.get(user_id)
        if not user:
            return jsonify({"error": "No data for this user"}), 404

        attentive_percent = (user["attentive_count"] / user["total_scans"] * 100) if user["total_scans"] > 0 else 0.0
        attentive_percent = round(attentive_percent, 1)
        
        df = pd.read_csv(LOCAL_FILE_PATH)
        last = df.tail(300)

        avg_temp = pd.to_numeric(last["Temperature"], errors='coerce').mean()
        avg_humidity = pd.to_numeric(last["Humidity"], errors='coerce').mean()
        avg_light = pd.to_numeric(last["Light Intensity"], errors='coerce').mean()
        avg_co2 = pd.to_numeric(last["Co2 Concentration"], errors='coerce').mean()

        door_status = last["Door Status"].mode().iloc[0] if not last["Door Status"].mode().empty else "Unknown"
        motion_status = last["Motion Status"].mode().iloc[0] if not last["Motion Status"].mode().empty else "Unknown"

        
        
        prompt = f"""
        You are a meeting attentiveness assistant. You receive participant metrics:

        - Attention percentage: {attentive_percent}%
        - Room conditions:
            • Temperature: {avg_temp:.1f}°C
            • Humidity: {avg_humidity:.1f}%
            • Light: {avg_light:.1f} lux
            • CO₂: {avg_co2:.0f} ppm
        - Room status:
            • Door: {door_status}
            • Motion: {motion_status}

        Task:
        Return exactly two clear, actionable bullet points under the heading "Suggestion:".

        - First bullet: attentiveness recommendation based only on the given attention percentage.  
        Use only these attentive activities: presenting (sitting or standing), typing on laptop, taking notes, writing on board, erasing board.  
        Guidance:  
            • If <50% → recommend 2–3 attentive activities.  
            • If 50–85% → recommend 1–2 activities.  
            • If >85% → recommend sustaining current focus with light reinforcement.  

        - Second bullet: environment recommendation.  
        If any condition exceeds comfort ranges (Temp >28°C, Humidity >70%, CO₂ >800 ppm, Light <150 lux) → suggest moving to a better-ventilated, cooler/brighter space as appropriate.  
        If all are within comfort ranges → write: "No environmental changes needed."  

        Rules:
        - Heading must be exactly "Suggestion:"  
        - Output must be only two bullets (no extras, no stars, no emojis, no long explanations).  
        - Keep total under 50 words.

        Format strictly:
        Suggestion:
        - <attentiveness suggestion>
        - <environment suggestion>
        """


        start = time.time()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = time.time() - start
        llm_tip = completion.choices[0].message.content
        logging.info(f"LLM call latency: {elapsed:.3f} seconds")

        plot_data = generate_ambient_plot(last, user_id)
        plot_file = plot_data["file_path"]
        plot_base64 = plot_data["base64"]

        final_suggestion = build_suggestion(
            attentive_percent, avg_temp, avg_humidity, avg_light, avg_co2,
            door_status, motion_status, llm_tip
        )

        # Productivity score calculation
        productivity_score = compute_meeting_productivity(
            [attentive_percent],
            {
                "light": avg_light,
                "temp": avg_temp,
                "humidity": avg_humidity,
                "co2": avg_co2
            }
        )
        
        # Individual user summary
        summary_row = {
            "timestamp": datetime.now().isoformat(),
            "attentive_percent": round(attentive_percent, 1),
            "avg_temp": round(avg_temp, 1),
            "avg_humidity": round(avg_humidity, 1),
            "avg_light": round(avg_light, 1),
            "avg_co2": round(avg_co2, 0),
            "door_status": door_status,
            "motion_status": motion_status,
            "productivity_A": productivity_score["A"],
            "productivity_E": productivity_score["E"],
            "productivity_MPS": productivity_score["MPS"],
            "suggestion": final_suggestion,
            "graph": plot_file
        }
        
        summary_file = f"{ATTENTION_LOG_DIR}/user_{user_id}_summary.csv"
        pd.DataFrame([summary_row]).to_csv(
            summary_file,
            index=False,
            mode='a',
            header=not os.path.exists(summary_file)
        )

        # Mark user ended & SNAPSHOT BEFORE reset
        users_data[user_id]["meeting_active"] = False
        meeting_state["ended_users"].add(user_id)
        meeting_state["snapshot"][user_id] = {
            "user_id": user_id,
            "attentive_count": users_data[user_id]["attentive_count"],
            "total_scans": users_data[user_id]["total_scans"],
            "attentive_percent": float(attentive_percent)
        }
        
        productive = bool(productivity_score["MPS"] >= 60)
        
        if productive:
            logging.info(f"🎉 Meeting for user {user_id} was PRODUCTIVE! MPS={productivity_score['MPS']:.1f}%")
        else:
            logging.info(f"⚠️ Meeting for user {user_id} was NOT productive. MPS={productivity_score['MPS']:.1f}%")

        # Reset this user's live counters (safe; snapshot kept)
        users_data[user_id] = {
            "last_status": "unknown",
            "attentive_count": 0,
            "total_scans": 0,
            "meeting_active": False
        }
        logging.info(f"✅ Reset counts for user {user_id} after meeting end.")

        # If all expected users ended, build FINAL owner summary from SNAPSHOT
        if meeting_state["expected_users"] and meeting_state["ended_users"] >= meeting_state["expected_users"]:
            logging.info("🎉 All expected users ended. Generating final owner report from snapshot...")
            summary_data = _collect_owner_summary_from_snapshot(meeting_state["snapshot"])
            summary_file2, plots = _save_owner_summary_files(summary_data)
            logging.info(f"✅ Final owner summary generated: overall={summary_data['overall_attentive_percent']}%, "
                         f"productive={summary_data['productive']}, file={summary_file2}, plots={plots}")

            # reset meeting state
            meeting_state["meeting_active"] = False
            meeting_state["start_time"] = None
            meeting_state["expected_users"].clear()
            meeting_state["ended_users"].clear()
            meeting_state["snapshot"].clear()

        resp = {
            "attentive_percent": float(round(attentive_percent, 1)),
            "avg_temp": float(round(avg_temp, 1)),
            "avg_humidity": float(round(avg_humidity, 1)),
            "avg_light": float(round(avg_light, 1)),
            "avg_co2": float(round(avg_co2, 0)),
            "door_status": str(door_status),
            "motion_status": str(motion_status),
            "productivity": {
                "A": float(productivity_score["A"]),
                "E": float(productivity_score["E"]),
                "MPS": float(productivity_score["MPS"]),
                "env_scores": {k: float(v) for k, v in productivity_score["env_scores"].items()}
            },
            "productive": productive,
            "suggestion": str(final_suggestion),
            "graph": str(plot_base64),
        }

        logging.info(f"End Meeting Response Payload Size : {len(str(resp))} bytes")
        return jsonify(resp)

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/owner_summary", methods=["GET"])
def owner_summary():
    try:
        summary_data = _collect_owner_summary_data()
        summary_file, plots = _save_owner_summary_files(summary_data)

        resp = {
            "overall_attentive_percent": summary_data["overall_attentive_percent"],
            "productive": summary_data["productive"],
            "productivity": summary_data["productivity"],  
            "summary_file": summary_file,
            "plots": plots,
            "users": summary_data["user_summaries"]
        }

        return jsonify(resp)
    except Exception as e:
        logging.error(f"❌ Error in /owner_summary: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)