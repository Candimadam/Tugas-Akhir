"""
app.py - Flask Application: YouTube Channel Recommendation System (Dual-Mode)
===============================================================================
Mendukung DUA mode input:
  Mode 1 — Channel → Channel:
    - Ambil embedding channel dari precomputed matrix
    - Hitung cosine similarity via precomputed similarity matrix (O(1))

  Mode 2 — Judul Video → Channel:
    - Preprocessing teks (text cleaning + lowercase)
    - Tokenisasi + inference IndoBERT fine-tuned (runtime embedding)
    - Mean pooling hidden state → L2 normalization
    - Hitung cosine similarity dengan semua channel embeddings (dot product)

Endpoints:
  GET  /                  → Halaman utama
  POST /recommend_channel → Mode 1 (channel → channel)
  POST /recommend_query   → Mode 2 (judul → channel)
  GET  /channels          → Daftar semua channel
  GET  /health            → Health check
"""

import json
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Konfigurasi Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path Konfigurasi
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Inisialisasi Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # Support karakter Unicode (Indonesia)

# ---------------------------------------------------------------------------
# Singleton: Data & Model yang di-load sekali saat startup
# ---------------------------------------------------------------------------
_data: dict = {}


def load_data() -> None:
    """
    Load semua artefak ke memory saat startup aplikasi.
    Meliputi: data channel, precomputed embeddings, model IndoBERT, tokenizer.
    """
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


# ---------------------------------------------------------------------------
# Preprocessing Teks (sama dengan pipeline notebook)
# ---------------------------------------------------------------------------

def text_cleaning(text: str) -> str:
    """
    Hapus emoji, simbol non-standar, dan karakter encoding rusak.
    Tanda baca standar (.,?!) dipertahankan agar sesuai pelatihan IndoBERT.
    """
    if not isinstance(text, str):
        return ""

    # Hapus emoji & simbol Unicode non-standar
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # simbol & piktogram
        "\U0001F680-\U0001F6FF"   # transport & peta
        "\U0001F1E0-\U0001F1FF"   # bendera
        "\U00002600-\U000026FF"   # simbol umum
        "\U00002700-\U000027BF"   # Dingbats
        "\U0001F900-\U0001F9FF"   # simbol tambahan
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002300-\U000023FF"   # teknis
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # Hapus karakter non-printable / encoding rusak
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", " ", text)

    # Hapus simbol dekoratif (pertahankan huruf, angka, tanda baca standar)
    text = re.sub(r"[^\w\s.,?!\-()/@#%+=\'\":;]", " ", text, flags=re.UNICODE)

    # Rapikan spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_query(text: str) -> str:
    """Pipeline preprocessing: text cleaning → lowercase."""
    return text_cleaning(text).lower()


# ---------------------------------------------------------------------------
# Embedding Functions (Mode 2)
# ---------------------------------------------------------------------------

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean pooling atas hidden states.

    Args:
        last_hidden_state : (batch, seq_len, hidden_size)
        attention_mask    : (batch, seq_len)   — 1 = token nyata, 0 = padding

    Returns:
        pooled            : (batch, hidden_size)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()   # (batch, seq, 1)
    summed        = (last_hidden_state * mask_expanded).sum(dim=1)
    counts        = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_text(text: str) -> np.ndarray:
    """
    Hasilkan embedding L2-normalized untuk satu teks input.

    Pipeline:
      1. Preprocessing (text_cleaning + lowercase)
      2. Tokenisasi IndoBERT
      3. Inference encoder (torch.no_grad)
      4. Mean pooling
      5. L2 normalization

    Returns:
        embedding: ndarray shape (768,)
    """
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


# ---------------------------------------------------------------------------
# Recommendation Functions
# ---------------------------------------------------------------------------

def get_channel_info(channel_name: str) -> dict:
    """Ambil metadata channel (kategori, subscriber, link)."""
    info = _data["channel_info"]
    if channel_name in info.index:
        row = info.loc[channel_name]
        return {
            "kategori":         row["kategori"],
            "jumlah_pelanggan": int(row["jumlah_pelanggan"]),
            "link_channel":     row["link_channel"],
        }
    return {"kategori": "-", "jumlah_pelanggan": 0, "link_channel": "#"}


def _build_results(top_k_scores: pd.Series) -> list[dict]:
    """Helper: ubah Series skor similarity → list dict hasil rekomendasi."""
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


def recommend_from_channel(
    channel_name: str,
    top_k: int = 5,
) -> Optional[list[dict]]:
    """
    Mode 1: Rekomendasi channel berdasarkan nama channel.

    Menggunakan precomputed similarity matrix → lookup O(1).
    Channel itu sendiri selalu dikecualikan dari hasil.

    Returns:
        List hasil rekomendasi, atau None jika channel tidak ditemukan.
    """
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


def recommend_from_query(
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Mode 2: Rekomendasi channel berdasarkan judul video bebas.

    Pipeline:
      1. embed_text(query_text)  → embedding L2-normalized (768,)
      2. Dot product dengan channel_matrix (sudah L2-normalized)
         → setara cosine similarity
      3. Ambil Top-K channel dengan skor tertinggi

    Returns:
        List hasil rekomendasi.
    Raises:
        ValueError jika teks kosong.
    """
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


# ---------------------------------------------------------------------------
# Helpers: parse payload
# ---------------------------------------------------------------------------

def _parse_payload() -> dict:
    """Dukung JSON body maupun form-data."""
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


def _parse_top_k(payload: dict, default: int = 5) -> int:
    try:
        return max(1, min(int(payload.get("top_k", default)), 10))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Halaman utama — tampilkan form dual-mode."""
    channel_names = _data.get("channel_names", [])
    ft_metadata   = _data.get("ft_metadata", {})
    return render_template(
        "index.html",
        channel_names=sorted(channel_names),
        ft_metadata=ft_metadata,
    )


@app.route("/recommend_channel", methods=["POST"])
def api_recommend_channel():
    """
    Mode 1: Rekomendasi channel → channel.

    Request JSON/form:
      - channel_name : str  (nama channel query)
      - top_k        : int  (default 5, max 20)

    Response JSON:
      - mode, query_channel, query_info, top_k, results, error
    """
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


@app.route("/recommend_query", methods=["POST"])
def api_recommend_query():
    """
    Mode 2: Rekomendasi channel berdasarkan judul video bebas.

    Request JSON/form:
      - query_text : str  (judul video yang dimasukkan user)
      - top_k      : int  (default 5, max 20)

    Response JSON:
      - mode, query_text, query_text_processed, top_k, results, error
    """
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


@app.route("/channels", methods=["GET"])
def list_channels():
    """Daftar semua channel yang tersedia."""
    channel_names = sorted(_data.get("channel_names", []))
    return jsonify({"channels": channel_names, "total": len(channel_names)})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status":         "ok",
        "total_channels": len(_data.get("channel_names", [])),
        "model":          _data.get("ft_metadata", {}).get("model_name", "unknown"),
        "modes":          ["channel", "query"],
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_data()
    app.run(debug=True, host="0.0.0.0", port=5000)
