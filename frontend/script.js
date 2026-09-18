"use strict";

/* ==========================================================================
   Constants
   ========================================================================== */
const API_BASE_URL = "http://localhost:8000";
const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const SUPPORTED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

const CLASS_LABELS = {
  Tomato___Early_blight: "Tomato Early Blight",
  Tomato___Late_blight: "Tomato Late Blight",
  Tomato___Leaf_Mold: "Tomato Leaf Mold",
  Tomato___healthy: "Healthy Tomato Leaf",
};

const HEALTHY_CLASS = "Tomato___healthy";

const TOAST_DURATION_MS = 4500;

/* ==========================================================================
   DOM references
   ========================================================================== */
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");

const heroCta = document.getElementById("heroCta");

const uploadZone = document.getElementById("uploadZone");
const chooseImageBtn = document.getElementById("chooseImageBtn");
const fileInput = document.getElementById("fileInput");

const previewZone = document.getElementById("previewZone");
const previewImage = document.getElementById("previewImage");
const previewFilename = document.getElementById("previewFilename");
const previewFilesize = document.getElementById("previewFilesize");
const removeImageBtn = document.getElementById("removeImageBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingZone = document.getElementById("loadingZone");

const resultZone = document.getElementById("resultZone");
const resultImage = document.getElementById("resultImage");
const resultDisease = document.getElementById("resultDisease");
const resultBadge = document.getElementById("resultBadge");
const resultDetails = document.querySelector(".result-details");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceTrack = document.getElementById("confidenceTrack");
const resetBtn = document.getElementById("resetBtn");

const toastContainer = document.getElementById("toastContainer");

/* ==========================================================================
   State
   ========================================================================== */
let selectedFile = null;
let isPredicting = false;

/* ==========================================================================
   Init
   ========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  registerEventListeners();
});

function registerEventListeners() {
  heroCta.addEventListener("click", () => {
    document.getElementById("predict").scrollIntoView({ behavior: "smooth" });
  });

  chooseImageBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
  });

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("is-dragover");
  });

  uploadZone.addEventListener("drop", handleDrop);

  fileInput.addEventListener("change", handleFileSelect);

  removeImageBtn.addEventListener("click", resetPrediction);
  analyzeBtn.addEventListener("click", predictDisease);
  resetBtn.addEventListener("click", resetPrediction);
}

/* ==========================================================================
   API health check
   ========================================================================== */
async function checkApiHealth() {
  try {
    const response = await fetch(HEALTH_ENDPOINT, { method: "GET" });

    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }

    setApiStatus(true);
  } catch (error) {
    setApiStatus(false);
  }
}

function setApiStatus(isOnline) {
  statusDot.classList.remove("status-dot--checking", "status-dot--online", "status-dot--offline");

  if (isOnline) {
    statusDot.classList.add("status-dot--online");
    statusLabel.textContent = "API Online";
  } else {
    statusDot.classList.add("status-dot--offline");
    statusLabel.textContent = "API Offline";
  }
}

/* ==========================================================================
   File selection & drag-and-drop
   ========================================================================== */
function handleFileSelect(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  processSelectedFile(file);
  fileInput.value = ""; // allow re-selecting the same file later
}

function handleDrop(event) {
  event.preventDefault();
  uploadZone.classList.remove("is-dragover");

  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (!file) return;
  processSelectedFile(file);
}

function processSelectedFile(file) {
  const validation = validateFile(file);

  if (!validation.valid) {
    showToast(validation.message, "error");
    return;
  }

  selectedFile = file;
  renderImagePreview(file);
}

/* ==========================================================================
   Validation
   ========================================================================== */
function validateFile(file) {
  if (!file) {
    return { valid: false, message: "No image selected. Please choose a file." };
  }

  if (!SUPPORTED_TYPES.includes(file.type)) {
    return {
      valid: false,
      message: "Unsupported file type. Please upload a JPG, JPEG, or PNG image.",
    };
  }

  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      message: "File is too large. Maximum allowed size is 10 MB.",
    };
  }

  return { valid: true, message: "" };
}

/* ==========================================================================
   Preview
   ========================================================================== */
function renderImagePreview(file) {
  const objectUrl = URL.createObjectURL(file);

  previewImage.src = objectUrl;
  previewFilename.textContent = file.name;
  previewFilesize.textContent = formatFileSize(file.size);

  showZone("preview");
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/* ==========================================================================
   Prediction
   ========================================================================== */
async function predictDisease() {
  if (isPredicting) return;

  if (!selectedFile) {
    showToast("No image selected. Please choose a file first.", "error");
    return;
  }

  isPredicting = true;
  setControlsDisabled(true);
  showZone("loading");

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Prediction request failed with status ${response.status}`);
    }

    const data = await response.json();
    validateAndDisplayPrediction(data);
  } catch (error) {
    handlePredictionError(error);
  } finally {
    isPredicting = false;
    setControlsDisabled(false);
  }
}

function validateAndDisplayPrediction(data) {
  if (
    !data ||
    typeof data.predicted_class !== "string" ||
    typeof data.confidence !== "number" ||
    Number.isNaN(data.confidence)
  ) {
    showToast("The server returned an unexpected response. Please try again.", "error");
    showZone("preview");
    return;
  }

  displayPrediction(data);
}

function handlePredictionError(error) {
  let message = "Prediction request failed. Please try again.";

  if (error instanceof TypeError) {
    // fetch throws a TypeError on network failure / CORS / API unreachable
    message = "Could not reach the API. Confirm the backend is running and reachable.";
  } else if (error && error.message) {
    message = error.message.includes("status")
      ? "The server could not process this image. Please try a different file."
      : error.message;
  }

  showToast(message, "error");
  showZone("preview");
}

/* ==========================================================================
   Result rendering
   ========================================================================== */
function displayPrediction(data) {
  const humanLabel = CLASS_LABELS[data.predicted_class] || data.predicted_class;
  const confidencePercent = Math.max(0, Math.min(100, data.confidence * 100));
  const isHealthy = data.predicted_class === HEALTHY_CLASS;

  resultImage.src = previewImage.src;
  resultDisease.textContent = humanLabel;
  confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
  confidenceTrack.setAttribute("aria-valuenow", confidencePercent.toFixed(1));

  resultDetails.classList.remove("is-healthy", "is-warning");
  resultDetails.classList.add(isHealthy ? "is-healthy" : "is-warning");
  resultBadge.textContent = isHealthy ? "Healthy Result" : "Disease Detected";

  showZone("result");

  // Animate the confidence bar after the result card is visible.
  confidenceFill.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${confidencePercent}%`;
    });
  });
}

/* ==========================================================================
   Reset
   ========================================================================== */
function resetPrediction() {
  selectedFile = null;

  if (previewImage.src && previewImage.src.startsWith("blob:")) {
    URL.revokeObjectURL(previewImage.src);
  }

  previewImage.src = "";
  previewFilename.textContent = "";
  previewFilesize.textContent = "";
  resultDisease.textContent = "—";
  confidenceValue.textContent = "0.0%";
  confidenceFill.style.width = "0%";
  resultDetails.classList.remove("is-healthy", "is-warning");

  showZone("upload");
}

/* ==========================================================================
   Zone / UI-state management
   ========================================================================== */
function showZone(zoneName) {
  const zones = {
    upload: uploadZone,
    preview: previewZone,
    loading: loadingZone,
    result: resultZone,
  };

  Object.entries(zones).forEach(([name, element]) => {
    element.hidden = name !== zoneName;
  });
}

function setControlsDisabled(disabled) {
  analyzeBtn.disabled = disabled;
  removeImageBtn.disabled = disabled;
  chooseImageBtn.disabled = disabled;
}

/* ==========================================================================
   Toast notifications
   ========================================================================== */
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast--${type}`;
  toast.setAttribute("role", "alert");

  const icon = document.createElement("span");
  icon.className = "toast-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = type === "error" ? "!" : "\u2713";

  const text = document.createElement("span");
  text.textContent = message;

  toast.appendChild(icon);
  toast.appendChild(text);
  toastContainer.appendChild(toast);

  window.setTimeout(() => {
    toast.classList.add("is-leaving");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  }, TOAST_DURATION_MS);
}
