/**
 * main.js — YouTube Frontend Logic
 * ===================================
 * - Mengelola pemilihan mode input dan submit ke endpoint yang sesuai
 * - Render hasil rekomendasi ke tabel
 * - Animasi loading, error handling, dan UI helpers
 */

"use strict";

/* ── DOM Refs ──────────────────────────────────────────── */
const tabChannel = document.getElementById("tab-channel");
const tabQuery = document.getElementById("tab-query");
const panelChannel = document.getElementById("panel-channel");
const panelQuery = document.getElementById("panel-query");
const modeHintText = document.getElementById("mode-hint-text");

const formChannel = document.getElementById("form-channel");
const formQuery = document.getElementById("form-query");

const channelSelect = document.getElementById("channel-select");
const topKChannel = document.getElementById("topk-channel");
const topKChannelDisplay = document.getElementById("topk-channel-display");
const btnChannel = document.getElementById("btn-channel");

const queryInput = document.getElementById("query-input");
const charCount = document.getElementById("char-count");
const topKQuery = document.getElementById("topk-query");
const topKQueryDisplay = document.getElementById("topk-query-display");
const btnQuery = document.getElementById("btn-query");

const loadingEl = document.getElementById("loading-indicator");
const errorBanner = document.getElementById("error-banner");
const errorMessageEl = document.getElementById("error-message");

const resultsSection = document.getElementById("results-section");
const resultModeBadge = document.getElementById("result-mode-badge");
const queryLink = document.getElementById("query-link");
const queryChannelName = document.getElementById("query-channel-name");
const queryKategori = document.getElementById("query-kategori");
const queryTextDisplay = document.getElementById("query-text-display");
const metaLabelText = document.getElementById("meta-label-text");
const metaValueText = document.getElementById("meta-value-text");
const resultTopK = document.getElementById("result-topk");
const resultCount = document.getElementById("result-count");
const resultsTbody = document.getElementById("results-tbody");

let activeMode = "channel";

/* ── Mode Switch ───────────────────────────────────────── */
function setActiveMode(mode) {
  activeMode = mode;

  const isChannel = mode === "channel";
  panelChannel.classList.toggle("is-hidden", !isChannel);
  panelQuery.classList.toggle("is-hidden", isChannel);

  tabChannel.classList.toggle("is-active", isChannel);
  tabQuery.classList.toggle("is-active", !isChannel);
  tabChannel.setAttribute("aria-selected", String(isChannel));
  tabQuery.setAttribute("aria-selected", String(!isChannel));

  modeHintText.textContent = isChannel
    ? "Kategori: Nama Channel"
    : "Kategori: Judul Video";
}

tabChannel.addEventListener("click", () => setActiveMode("channel"));
tabQuery.addEventListener("click", () => setActiveMode("query"));

/* ── Live helpers ──────────────────────────────────────── */
topKChannel.addEventListener("input", () => {
  topKChannelDisplay.textContent = topKChannel.value;
});

topKQuery.addEventListener("input", () => {
  topKQueryDisplay.textContent = topKQuery.value;
});

queryInput.addEventListener("input", () => {
  charCount.textContent = String(queryInput.value.length);
});

setActiveMode("channel");

/* ── Form Submit ───────────────────────────────────────── */
formChannel.addEventListener("submit", async (e) => {
  e.preventDefault();

  const channelName = channelSelect.value.trim();
  const topK = parseInt(topKChannel.value, 10);

  if (!channelName) {
    showError("Silakan pilih channel terlebih dahulu.");
    return;
  }

  setLoadingState(true);
  clearError();
  hideResults();

  try {
    const response = await fetch("/recommend_channel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channel_name: channelName, top_k: topK }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || `Server error: ${response.status}`);
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Gagal terhubung ke server. Pastikan Flask sudah berjalan.");
    console.error("[YouTube] Fetch error:", err);
  } finally {
    setLoadingState(false);
  }
});

formQuery.addEventListener("submit", async (e) => {
  e.preventDefault();

  const queryText = queryInput.value.trim();
  const topK = parseInt(topKQuery.value, 10);

  if (!queryText) {
    showError("Silakan masukkan judul video terlebih dahulu.");
    return;
  }

  setLoadingState(true);
  clearError();
  hideResults();

  try {
    const response = await fetch("/recommend_query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_text: queryText, top_k: topK }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || `Server error: ${response.status}`);
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Gagal terhubung ke server. Pastikan Flask sudah berjalan.");
    console.error("[YouTube] Fetch error:", err);
  } finally {
    setLoadingState(false);
  }
});

/* ── State Helpers ─────────────────────────────────────── */
function setLoadingState(isLoading) {
  loadingEl.hidden = !isLoading;
  btnChannel.disabled = isLoading;
  btnQuery.disabled = isLoading;

  const channelText = btnChannel.querySelector(".btn__text");
  const queryText = btnQuery.querySelector(".btn__text");

  if (isLoading) {
    channelText.textContent = "Menghitung…";
    queryText.textContent = "Menghitung…";
  } else {
    channelText.textContent = "Rekomendasikan";
    queryText.textContent = "Cari Channel Relevan";
  }
}

function showError(message) {
  errorMessageEl.textContent = message;
  errorBanner.hidden = false;
  errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearError() {
  errorBanner.hidden = true;
  errorMessageEl.textContent = "";
}

function hideResults() {
  resultsSection.hidden = true;
}

/* ── Render Results ────────────────────────────────────── */
function renderResults(data) {
  const isChannelMode = data.mode === "channel";
  const topK = data.top_k;
  const results = data.results || [];

  resultModeBadge.textContent = isChannelMode
    ? "Kategori Channel"
    : "Kategori Judul";
  resultTopK.textContent = topK;
  resultCount.textContent = `${results.length} hasil ditemukan`;

  queryLink.hidden = !isChannelMode;
  queryTextDisplay.hidden = isChannelMode;
  queryKategori.hidden = false;

  if (isChannelMode) {
    const queryInfo = data.query_info || {};
    queryChannelName.textContent = data.query_channel || "-";
    queryLink.href = queryInfo.link_channel || "#";
    queryKategori.textContent = queryInfo.kategori || "-";
    metaLabelText.textContent = "Subscriber";
    metaValueText.textContent = formatSubscribers(queryInfo.jumlah_pelanggan);
  } else {
    queryTextDisplay.textContent = data.query_text || "-";
    queryKategori.textContent = "Judul Video";
    metaLabelText.textContent = "Teks diproses";
    metaValueText.textContent = data.query_text_processed || "-";
    queryLink.hidden = true;
  }

  // Build table rows
  resultsTbody.innerHTML = "";

  results.forEach((item, idx) => {
    const row = document.createElement("tr");
    // Stagger animation delay
    row.style.animationDelay = `${idx * 55}ms`;

    row.innerHTML = `
      <td class="td-rank">
        <span class="td-rank-badge ${rankClass(item.rank)}">${item.rank}</span>
      </td>
      <td class="td-channel">
        <a href="${escapeHTML(item.link_channel)}" target="_blank" rel="noopener">
          ${escapeHTML(item.nama_channel)}
        </a>
      </td>
      <td>
        <span class="badge badge-kategori">${escapeHTML(item.kategori)}</span>
      </td>
      <td class="td-subscriber">
        ${formatSubscribers(item.jumlah_pelanggan)}
      </td>
      <td>
        <div class="score-wrap">
          <span class="score-value">${item.similarity_score.toFixed(4)}</span>
          <div class="score-bar-bg">
            <div
              class="score-bar-fill"
              style="width: ${(item.similarity_score * 100).toFixed(1)}%"
            ></div>
          </div>
        </div>
      </td>
    `;

    resultsTbody.appendChild(row);
  });

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── Utility Functions ─────────────────────────────────── */
/**
 * Format angka subscriber ke format pendek (K / Jt).
 * @param {number} num
 * @returns {string}
 */
function formatSubscribers(num) {
  if (!num || isNaN(num)) return "-";
  if (num >= 1_000_000)
    return (num / 1_000_000).toFixed(1).replace(".0", "") + " Jt";
  if (num >= 1_000) return (num / 1_000).toFixed(1).replace(".0", "") + " K";
  return num.toString();
}

/**
 * Pilih CSS class badge berdasarkan ranking.
 * @param {number} rank
 * @returns {string}
 */
function rankClass(rank) {
  if (rank === 1) return "rank-gold";
  if (rank === 2) return "rank-silver";
  if (rank === 3) return "rank-bronze";
  return "rank-normal";
}

/**
 * Escape HTML untuk menghindari XSS pada konten dinamis.
 * @param {string} str
 * @returns {string}
 */
function escapeHTML(str) {
  if (typeof str !== "string") return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
