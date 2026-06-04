import json
import logging
import os
import re
import emoji
from typing import Optional

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# KONFIGURASI LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# PATH KONFIGURASI
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(
    PROJECT_DIR, "output", "fine_tuning_experiment", "model_fine_tuned"
)

DATA_PATHS = {
    # Channel embedding matrix (100 x 768), sudah L2-normalized dari notebook
    "channel_matrix": os.path.join(
        PROJECT_DIR, "output", "fine_tuning_experiment", "channel_matrix_fine_tuned.npy"
    ),
    # Precomputed similarity matrix (100 x 100)
    "similarity_matrix": os.path.join(
        PROJECT_DIR, "output", "fine_tuning_experiment", "similarity_matrix_fine_tuned.csv"
    ),
    # Nama channel (urutan sesuai baris channel_matrix)
    "channel_names": os.path.join(PROJECT_DIR, "output", "channel_names.csv"),
    # Metadata video untuk info tambahan (kategori, subscriber, link)
    "video_metadata": os.path.join(PROJECT_DIR, "output", "video_metadata.csv"),
    # Metadata fine-tuning
    "metadata": os.path.join(
        PROJECT_DIR, "output", "fine_tuning_experiment", "metadata.json"
    ),
}

# INISIALISASI FLASK
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # Support karakter Unicode (Indonesia)

# SINGLETON: DATA & MODEL DI-LOAD SEKALI SAAT STARTUP
_data: dict = {}

# Load semua data dan model ke memory saat startup aplikasi
def load_data() -> None:
    global _data

    logger.info("Memuat data dan model...")

    # ── 1. Channel names ───────────────────────────────────────────────────
    channel_names_df = pd.read_csv(DATA_PATHS["channel_names"])
    channel_names: list[str] = channel_names_df["nama_channel"].tolist()

    # ── 2. Channel embedding matrix (100 x 768) ────────────────────────────
    channel_matrix = np.load(DATA_PATHS["channel_matrix"]).astype(np.float32)
    channel_matrix = normalize(channel_matrix, norm="l2")   # pastikan L2-normalized

    # ── 3. Precomputed similarity matrix ───────────────────────────────────
    similarity_df = pd.read_csv(DATA_PATHS["similarity_matrix"], index_col=0)

    # ── 4. Metadata channel ────────────────────────────────────────────────
    video_metadata = pd.read_csv(DATA_PATHS["video_metadata"])
    channel_info = (
        video_metadata.drop_duplicates(subset="nama_channel")
        .set_index("nama_channel")[["kategori", "jumlah_pelanggan", "link_channel"]]
    )

    # ── 5. Metadata fine-tuning ────────────────────────────────────────────
    with open(DATA_PATHS["metadata"], "r", encoding="utf-8") as f:
        ft_metadata = json.load(f)

    # ── 6. Load IndoBERT fine-tuned (untuk Mode 2) ─────────────────────────
    logger.info("Memuat model IndoBERT dari: %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Load model klasifikasi → ambil encoder (base_model) sebagai feature extractor
    clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    encoder   = clf_model.base_model    # BertModel (tanpa classification head)
    encoder.eval()                      # Mode inference, matikan dropout

    device = torch.device("cpu")        # CPU-only (kompatibel semua environment)
    encoder.to(device)

    _data = {
        "channel_names":  channel_names,
        "channel_matrix": channel_matrix,   # ndarray (100, 768)
        "similarity_df":  similarity_df,
        "channel_info":   channel_info,
        "ft_metadata":    ft_metadata,
        "tokenizer":      tokenizer,
        "encoder":        encoder,
        "device":         device,
        "max_length":     ft_metadata.get("max_length", 128),
    }

    logger.info(
        "Siap. Total channel: %d | Dimensi embedding: %d | Device: %s",
        len(channel_names),
        channel_matrix.shape[1],
        device,
    )

# PREPROCESSING TEKS (SAMA DENGAN PIPELINE DI NOTEBOOK)
def text_cleaning(text: str) -> str:
    # Hapus emoji, karakter kontrol, dan simbol dekoratif yang tidak dibutuhkan
    text = emoji.replace_emoji(text, replace=' ')
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]|[^\w\s.,?!\-()/@#%+=\'":;]', ' ', text, flags=re.UNICODE)

    # Rapikan spasi berlebih di akhir proses cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_query(text: str) -> str:
    """Pipeline preprocessing: text cleaning → lowercase."""
    return text_cleaning(text).lower()

# EMBEDDING FUNCTIONS (MODE 2)
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).float()   # (batch, seq, 1)
    summed        = (last_hidden_state * mask_expanded).sum(dim=1)
    counts        = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_text(text: str) -> np.ndarray:
    tokenizer  = _data["tokenizer"]
    encoder    = _data["encoder"]
    device     = _data["device"]
    max_length = _data["max_length"]

    # Preprocessing
    processed = preprocess_query(text)
    if not processed:
        raise ValueError("Teks input kosong setelah preprocessing.")

    # Tokenisasi
    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference (tanpa gradient untuk efisiensi)
    with torch.no_grad():
        outputs = encoder(**inputs)

    # Mean pooling → L2 normalization
    pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    pooled_np = pooled.cpu().numpy().astype(np.float32)      # shape (1, 768)
    return normalize(pooled_np, norm="l2")[0]                 # shape (768,)

# RECOMMENDATION FUNCTIONS
def get_channel_info(channel_name: str) -> dict:
    info = _data["channel_info"]
    if channel_name in info.index:
        row = info.loc[channel_name]
        return {
            "kategori":         row["kategori"],
            "jumlah_pelanggan": int(row["jumlah_pelanggan"]),
            "link_channel":     row["link_channel"],
        }
    return {"kategori": "-", "jumlah_pelanggan": 0, "link_channel": "#"}

# Bantu bangun response hasil rekomendasi untuk kedua mode (Mode 1 & Mode 2)
def _build_results(top_k_scores: pd.Series) -> list[dict]:
    results = []
    for rank, (ch_name, score) in enumerate(top_k_scores.items(), start=1):
        info = get_channel_info(ch_name)
        results.append({
            "rank":             rank,
            "nama_channel":     ch_name,
            "similarity_score": round(float(score), 4),
            "kategori":         info["kategori"],
            "jumlah_pelanggan": info["jumlah_pelanggan"],
            "link_channel":     info["link_channel"],
        })
    return results

# Mode 1: Rekomendasi berdasarkan nama channel (lookup similarity matrix)
def recommend_from_channel(
    channel_name: str,
    top_k: int = 5,
) -> Optional[list[dict]]:
    similarity_df: pd.DataFrame = _data["similarity_df"]

    if channel_name not in similarity_df.index:
        logger.warning("Channel '%s' tidak ditemukan di similarity matrix.", channel_name)
        return None

    sim_scores = (
        similarity_df.loc[channel_name]
        .drop(labels=channel_name, errors="ignore")
        .sort_values(ascending=False)
        .head(top_k)
    )
    return _build_results(sim_scores)

# Mode 2: Rekomendasi berdasarkan judul video bebas (embedding + dot product)
def recommend_from_query(
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    channel_matrix: np.ndarray = _data["channel_matrix"]   # (100, 768)
    channel_names:  list[str]  = _data["channel_names"]

    # Generate embedding query secara realtime
    query_emb = embed_text(query_text)     # (768,)

    # Cosine similarity = dot product (karena keduanya L2-normalized)
    scores = channel_matrix.dot(query_emb)  # (100,)

    # Ambil Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores  = pd.Series(
        {channel_names[i]: scores[i] for i in top_indices}
    )
    return _build_results(top_scores)

# HELPERS: PARSE PAYLOAD
def _parse_payload() -> dict:
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()

# HELPERS: VALIDASI TOP-K
def _parse_top_k(payload: dict, default: int = 5) -> int:
    try:
        return max(1, min(int(payload.get("top_k", default)), 10))
    except (ValueError, TypeError):
        return default

# ROUTES: Halaman utama dengan form dual-mode
@app.route("/", methods=["GET"])
def index():
    channel_names = _data.get("channel_names", [])
    ft_metadata   = _data.get("ft_metadata", {})
    return render_template(
        "index.html",
        channel_names=sorted(channel_names),
        ft_metadata=ft_metadata,
    )

# API endpoint untuk rekomendasi channel berdasarkan nama channel (Mode 1)
@app.route("/recommend_channel", methods=["POST"])
def api_recommend_channel():
    payload      = _parse_payload()
    channel_name = payload.get("channel_name", "").strip()
    top_k        = _parse_top_k(payload)

    if not channel_name:
        return jsonify({"error": "Nama channel tidak boleh kosong.", "results": []}), 400

    results = recommend_from_channel(channel_name, top_k=top_k)
    if results is None:
        return jsonify({
            "error":   f"Channel '{channel_name}' tidak ditemukan dalam database.",
            "results": [],
        }), 404

    query_info = get_channel_info(channel_name)
    return jsonify({
        "mode":          "channel",
        "query_channel": channel_name,
        "query_info":    query_info,
        "top_k":         top_k,
        "results":       results,
        "error":         None,
    })

# API endpoint untuk rekomendasi channel berdasarkan judul video bebas (Mode 2)
@app.route("/recommend_query", methods=["POST"])
def api_recommend_query():
    payload    = _parse_payload()
    query_text = payload.get("query_text", "").strip()
    top_k      = _parse_top_k(payload)

    if not query_text:
        return jsonify({"error": "Judul video tidak boleh kosong.", "results": []}), 400

    try:
        results = recommend_from_query(query_text, top_k=top_k)
    except ValueError as e:
        return jsonify({"error": str(e), "results": []}), 400
    except Exception as e:
        logger.exception("Error saat generate embedding query.")
        return jsonify({"error": "Gagal memproses teks input.", "results": []}), 500

    processed = preprocess_query(query_text)
    return jsonify({
        "mode":                 "query",
        "query_text":           query_text,
        "query_text_processed": processed,
        "top_k":                top_k,
        "results":              results,
        "error":                None,
    })

# API endpoint untuk daftar semua channel (untuk dropdown di frontend)
@app.route("/channels", methods=["GET"])
def list_channels():
    channel_names = sorted(_data.get("channel_names", []))
    return jsonify({"channels": channel_names, "total": len(channel_names)})

# API endpoint untuk health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "total_channels": len(_data.get("channel_names", [])),
        "model":          _data.get("ft_metadata", {}).get("model_name", "unknown"),
        "modes":          ["channel", "query"],
    })

# ENTRYPOINT
if __name__ == "__main__":
    load_data()
    app.run(debug=True, host="0.0.0.0", port=5000)
