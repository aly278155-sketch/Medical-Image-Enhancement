import { useState, useRef, useCallback } from "react";

// ─── Design tokens ───────────────────────────────────────────────────────────
const C = {
  bg: "#0a0e17",
  surface: "#111827",
  card: "#161d2e",
  border: "#1e2d45",
  accent: "#00d4ff",
  accentDim: "#0099bb",
  green: "#00e676",
  amber: "#ffb300",
  red: "#ff4444",
  text: "#e2e8f0",
  muted: "#64748b",
  white: "#ffffff",
};

// ─── Fake OpenCV simulation (runs in browser) ────────────────────────────────
function applyTechnique(canvas, technique, params) {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const w = canvas.width, h = canvas.height;

  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) gray[i] = data[i * 4];

  let result = new Float32Array(gray);

  if (technique === "clahe") {
    // Simplified CLAHE simulation
    const clipLimit = params.clipLimit || 2.5;
    const tileSize = params.tileSize || 8;
    const tilesX = Math.ceil(w / tileSize);
    const tilesY = Math.ceil(h / tileSize);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const x0 = tx * tileSize, y0 = ty * tileSize;
        const x1 = Math.min(x0 + tileSize, w);
        const y1 = Math.min(y0 + tileSize, h);
        const hist = new Float32Array(256);
        let count = 0;
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++) { hist[Math.round(gray[y * w + x])]++; count++; }
        const limit = (clipLimit * count) / 256;
        let excess = 0;
        for (let i = 0; i < 256; i++) { if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit; } }
        const add = excess / 256;
        for (let i = 0; i < 256; i++) hist[i] += add;
        const cdf = new Float32Array(256);
        cdf[0] = hist[0]; for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];
        const cdfMin = cdf.find(v => v > 0) || 1;
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++) {
            const v = Math.round(gray[y * w + x]);
            result[y * w + x] = ((cdf[v] - cdfMin) / (count - cdfMin)) * 255;
          }
      }
    }
  } else if (technique === "histogram_eq") {
    const hist = new Float32Array(256);
    for (let i = 0; i < gray.length; i++) hist[Math.round(gray[i])]++;
    const cdf = new Float32Array(256);
    cdf[0] = hist[0]; for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];
    const cdfMin = cdf.find(v => v > 0) || 1;
    const total = w * h;
    for (let i = 0; i < gray.length; i++) {
      const v = Math.round(gray[i]);
      result[i] = ((cdf[v] - cdfMin) / (total - cdfMin)) * 255;
    }
  } else if (technique === "gamma") {
    const gamma = params.gamma || 1.4;
    for (let i = 0; i < gray.length; i++)
      result[i] = Math.pow(gray[i] / 255, 1 / gamma) * 255;
  } else if (technique === "gaussian") {
    const k = params.kernelSize || 5;
    const sigma = params.sigma || 1.0;
    const kernel = buildGaussianKernel(k, sigma);
    result = convolve(gray, w, h, kernel, k);
  } else if (technique === "median") {
    const k = params.kernelSize || 5;
    result = medianFilter(gray, w, h, k);
  } else if (technique === "bilateral") {
    result = bilateralFilter(gray, w, h, 5, params.sigmaColor || 60, params.sigmaSpace || 60);
  } else if (technique === "canny") {
    const blurred = convolve(gray, w, h, buildGaussianKernel(5, 1.0), 5);
    result = cannyEdges(blurred, w, h, params.low || 30, params.high || 120);
  } else if (technique === "sobel") {
    result = sobelEdges(gray, w, h);
  } else if (technique === "unsharp") {
    const strength = params.strength || 1.5;
    const blurred = convolve(gray, w, h, buildGaussianKernel(5, 1.0), 5);
    for (let i = 0; i < gray.length; i++)
      result[i] = Math.min(255, Math.max(0, gray[i] * (1 + strength) - blurred[i] * strength));
  }

  // Write result back
  const out = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.min(255, Math.max(0, Math.round(result[i])));
    out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = v;
    out.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);

  // Compute stats
  const vals = Array.from({ length: w * h }, (_, i) => result[i]);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const std = Math.sqrt(vals.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / vals.length);
  const snr = std > 0 ? mean / std : 0;
  return { snr: snr.toFixed(2), contrast: std.toFixed(2), mean: mean.toFixed(1) };
}

function buildGaussianKernel(size, sigma) {
  const k = Math.floor(size / 2);
  const kernel = [];
  let sum = 0;
  for (let y = -k; y <= k; y++) for (let x = -k; x <= k; x++) {
    const v = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
    kernel.push(v); sum += v;
  }
  return kernel.map(v => v / sum);
}

function convolve(data, w, h, kernel, kSize) {
  const k = Math.floor(kSize / 2);
  const result = new Float32Array(data.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let ky = -k; ky <= k; ky++) for (let kx = -k; kx <= k; kx++) {
      const ny = Math.min(h - 1, Math.max(0, y + ky));
      const nx = Math.min(w - 1, Math.max(0, x + kx));
      sum += data[ny * w + nx] * kernel[(ky + k) * kSize + (kx + k)];
    }
    result[y * w + x] = sum;
  }
  return result;
}

function medianFilter(data, w, h, kSize) {
  const k = Math.floor(kSize / 2);
  const result = new Float32Array(data.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const neighborhood = [];
    for (let ky = -k; ky <= k; ky++) for (let kx = -k; kx <= k; kx++) {
      const ny = Math.min(h - 1, Math.max(0, y + ky));
      const nx = Math.min(w - 1, Math.max(0, x + kx));
      neighborhood.push(data[ny * w + nx]);
    }
    neighborhood.sort((a, b) => a - b);
    result[y * w + x] = neighborhood[Math.floor(neighborhood.length / 2)];
  }
  return result;
}

function bilateralFilter(data, w, h, d, sigmaColor, sigmaSpace) {
  const k = Math.floor(d / 2);
  const result = new Float32Array(data.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let sum = 0, wSum = 0;
    const center = data[y * w + x];
    for (let ky = -k; ky <= k; ky++) for (let kx = -k; kx <= k; kx++) {
      const ny = Math.min(h - 1, Math.max(0, y + ky));
      const nx = Math.min(w - 1, Math.max(0, x + kx));
      const val = data[ny * w + nx];
      const spaceDist = kx * kx + ky * ky;
      const colorDist = (val - center) ** 2;
      const weight = Math.exp(-spaceDist / (2 * sigmaSpace * sigmaSpace) - colorDist / (2 * sigmaColor * sigmaColor));
      sum += val * weight; wSum += weight;
    }
    result[y * w + x] = sum / wSum;
  }
  return result;
}

function sobelEdges(data, w, h) {
  const result = new Float32Array(data.length);
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let gx = 0, gy = 0;
    for (let ky2 = -1; ky2 <= 1; ky2++) for (let kx2 = -1; kx2 <= 1; kx2++) {
      const v = data[(y + ky2) * w + (x + kx2)];
      gx += v * kx[(ky2 + 1) * 3 + (kx2 + 1)];
      gy += v * ky[(ky2 + 1) * 3 + (kx2 + 1)];
    }
    result[y * w + x] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
  }
  return result;
}

function cannyEdges(data, w, h, lowT, highT) {
  const sobel = sobelEdges(data, w, h);
  const result = new Float32Array(data.length);
  for (let i = 0; i < sobel.length; i++)
    result[i] = sobel[i] >= highT ? 255 : sobel[i] >= lowT ? 128 : 0;
  return result;
}

// ─── Synthetic X-Ray Generator ───────────────────────────────────────────────
function generateSyntheticXRay(canvas) {
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  const imageData = ctx.createImageData(w, h);
  const data = imageData.data;
  const cx = w / 2, cy = h / 2;

  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
    let v = Math.max(0, 1 - dist / (Math.min(w, h) * 0.55)) * 200;

    // Add circular structures
    const structures = [
      { x: w * 0.35, y: h * 0.4, r: w * 0.12, i: 55 },
      { x: w * 0.65, y: h * 0.4, r: w * 0.12, i: 55 },
      { x: w * 0.5, y: h * 0.55, r: w * 0.06, i: 40 },
      { x: w * 0.42, y: h * 0.35, r: w * 0.035, i: 70 },
      { x: w * 0.58, y: h * 0.45, r: w * 0.025, i: 70 },
    ];
    for (const s of structures) {
      const d = Math.sqrt((x - s.x) ** 2 + (y - s.y) ** 2);
      if (d < s.r) v = Math.min(255, v + s.i * (1 - d / s.r));
    }

    // Gaussian noise
    v += (Math.random() - 0.5) * 35;
    v = Math.min(255, Math.max(0, Math.round(v)));
    const i = (y * w + x) * 4;
    data[i] = data[i + 1] = data[i + 2] = v; data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

// ─── UI Components ────────────────────────────────────────────────────────────
const Badge = ({ label, color = C.accent }) => (
  <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 99,
    background: color + "22", color, border: `1px solid ${color}44`, letterSpacing: 1 }}>
    {label}
  </span>
);

const StatBox = ({ label, value, unit = "" }) => (
  <div style={{ textAlign: "center", padding: "8px 12px", background: C.bg,
    borderRadius: 8, border: `1px solid ${C.border}` }}>
    <div style={{ fontSize: 18, fontWeight: 700, color: C.accent, fontFamily: "monospace" }}>
      {value}<span style={{ fontSize: 11, color: C.muted }}>{unit}</span>
    </div>
    <div style={{ fontSize: 10, color: C.muted, marginTop: 2, letterSpacing: 0.5 }}>{label}</div>
  </div>
);

const TechButton = ({ id, label, tag, active, onClick }) => (
  <button onClick={() => onClick(id)} style={{
    padding: "8px 14px", borderRadius: 8, border: `1px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent + "18" : C.card, color: active ? C.accent : C.text,
    cursor: "pointer", fontSize: 12, fontWeight: active ? 700 : 400,
    display: "flex", flexDirection: "column", gap: 3, alignItems: "flex-start",
    transition: "all 0.15s", minWidth: 110
  }}>
    <span>{label}</span>
    {tag && <Badge label={tag} color={active ? C.accent : C.muted} />}
  </button>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
const TECHNIQUES = [
  { id: "original", label: "Original", tag: "BASE", group: "base" },
  { id: "clahe", label: "CLAHE", tag: "CONTRAST", group: "contrast" },
  { id: "histogram_eq", label: "Hist. EQ", tag: "CONTRAST", group: "contrast" },
  { id: "gamma", label: "Gamma γ", tag: "CONTRAST", group: "contrast" },
  { id: "gaussian", label: "Gaussian", tag: "DENOISE", group: "noise" },
  { id: "median", label: "Median", tag: "DENOISE", group: "noise" },
  { id: "bilateral", label: "Bilateral", tag: "DENOISE", group: "noise" },
  { id: "canny", label: "Canny", tag: "EDGE", group: "edge" },
  { id: "sobel", label: "Sobel", tag: "EDGE", group: "edge" },
  { id: "unsharp", label: "Unsharp", tag: "SHARPEN", group: "edge" },
];

const GROUP_COLORS = {
  base: C.muted, contrast: C.accent, noise: C.green, edge: C.amber
};

const PRESETS = [
  { id: "xray", label: "X-Ray", icon: "🦴", steps: ["clahe", "bilateral", "canny"] },
  { id: "mri", label: "MRI", icon: "🧠", steps: ["gaussian", "clahe", "unsharp"] },
  { id: "ct", label: "CT Scan", icon: "🫁", steps: ["gaussian", "gamma", "sobel"] },
  { id: "mammo", label: "Mammography", icon: "🔬", steps: ["median", "clahe"] },
];

export default function App() {
  const originalRef = useRef(null);
  const processedRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [active, setActive] = useState("original");
  const [stats, setStats] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [history, setHistory] = useState([]);
  const [params, setParams] = useState({
    clipLimit: 2.5, tileSize: 8, gamma: 1.4, kernelSize: 5,
    sigma: 1.0, sigmaColor: 60, sigmaSpace: 60, low: 30, high: 120, strength: 1.5
  });

  const initCanvases = useCallback(() => {
    const orig = originalRef.current;
    const proc = processedRef.current;
    if (!orig || !proc) return;
    orig.width = proc.width = 384;
    orig.height = proc.height = 384;
    generateSyntheticXRay(orig);
    const ctx2 = proc.getContext("2d");
    ctx2.drawImage(orig, 0, 0);
    setLoaded(true);
    setActive("original");
    setStats(null);
    setHistory([]);
  }, []);

  const handleUpload = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const orig = originalRef.current;
        const proc = processedRef.current;
        orig.width = proc.width = 384;
        orig.height = proc.height = 384;
        const ctx = orig.getContext("2d");
        ctx.drawImage(img, 0, 0, 384, 384);
        // Convert to grayscale
        const id = ctx.getImageData(0, 0, 384, 384);
        for (let i = 0; i < id.data.length; i += 4) {
          const g = Math.round(0.299 * id.data[i] + 0.587 * id.data[i + 1] + 0.114 * id.data[i + 2]);
          id.data[i] = id.data[i + 1] = id.data[i + 2] = g;
        }
        ctx.putImageData(id, 0, 0);
        proc.getContext("2d").drawImage(orig, 0, 0);
        setLoaded(true); setActive("original"); setStats(null); setHistory([]);
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }, []);

  const applyTech = useCallback((techId) => {
    if (!loaded || techId === "original") {
      if (loaded && techId === "original") {
        const orig = originalRef.current;
        const proc = processedRef.current;
        proc.getContext("2d").drawImage(orig, 0, 0);
        setActive("original"); setStats(null);
      }
      return;
    }
    setProcessing(true);
    setTimeout(() => {
      try {
        const orig = originalRef.current;
        const proc = processedRef.current;
        proc.getContext("2d").drawImage(orig, 0, 0);
        const s = applyTechnique(proc, techId, params);
        setStats(s);
        setActive(techId);
        setHistory(h => [...h.slice(-4), techId]);
      } finally {
        setProcessing(false);
      }
    }, 30);
  }, [loaded, params]);

  const applyPreset = useCallback((preset) => {
    if (!loaded) return;
    setProcessing(true);
    const proc = processedRef.current;
    proc.getContext("2d").drawImage(originalRef.current, 0, 0);
    let lastStats = null;
    for (const step of preset.steps) {
      lastStats = applyTechnique(proc, step, params);
    }
    setStats(lastStats);
    setActive(preset.steps[preset.steps.length - 1]);
    setHistory(preset.steps);
    setProcessing(false);
  }, [loaded, params]);

  const downloadResult = useCallback(() => {
    const link = document.createElement("a");
    link.download = `enhanced_${active}.png`;
    link.href = processedRef.current.toDataURL();
    link.click();
  }, [active]);

  const groupLabel = { base: "Base", contrast: "Contrast", noise: "Noise Reduction", edge: "Edge / Sharpen" };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'Segoe UI', system-ui, sans-serif", padding: "0 0 40px" }}>

      {/* ── Header ── */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: "14px 24px", display: "flex", alignItems: "center", gap: 14 }}>
        <span style={{ fontSize: 28 }}>🏥</span>
        <div>
          <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: -0.5, color: C.white }}>
            Medical Image Enhancement System
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 1 }}>
            نظام تحسين الصور الطبية الإشعاعية · OpenCV Techniques · Python-based
          </div>
          <div style={{ fontSize: 11, color: C.accent, marginTop: 4, fontWeight: 600 }}>
            👨‍💻 Ali Hussein Allawi &nbsp;|&nbsp; Medical Physics &nbsp;|&nbsp; University of Al-Qadisiyah, Iraq
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <Badge label="CLAHE" color={C.accent} />
          <Badge label="DENOISE" color={C.green} />
          <Badge label="EDGE DETECTION" color={C.amber} />
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 24px 0" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20 }}>

          {/* ── Left: Canvases ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Load controls */}
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <button onClick={initCanvases} style={{
                padding: "9px 18px", background: C.accent, color: C.bg,
                border: "none", borderRadius: 8, fontWeight: 700, cursor: "pointer", fontSize: 13
              }}>⚡ Generate X-Ray Sample</button>
              <label style={{ padding: "9px 18px", background: C.card, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: 8, fontWeight: 600,
                cursor: "pointer", fontSize: 13 }}>
                📁 Upload Image
                <input type="file" accept="image/*" onChange={handleUpload} style={{ display: "none" }} />
              </label>
              {loaded && (
                <button onClick={downloadResult} style={{
                  padding: "9px 18px", background: C.card, color: C.green,
                  border: `1px solid ${C.green}44`, borderRadius: 8, fontWeight: 600,
                  cursor: "pointer", fontSize: 13 }}>
                  ⬇ Download Result
                </button>
              )}
            </div>

            {/* Canvas row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                { label: "Original Image", ref: originalRef, badge: "INPUT", badgeColor: C.muted },
                { label: active === "original" ? "No Filter Applied" : `Enhanced · ${active.toUpperCase()}`,
                  ref: processedRef, badge: active === "original" ? "PREVIEW" : "OUTPUT", badgeColor: C.accent }
              ].map(({ label, ref, badge, badgeColor }) => (
                <div key={badge} style={{ background: C.card, borderRadius: 12,
                  border: `1px solid ${C.border}`, overflow: "hidden" }}>
                  <div style={{ padding: "10px 14px", borderBottom: `1px solid ${C.border}`,
                    display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: C.text }}>{label}</span>
                    <Badge label={badge} color={badgeColor} />
                  </div>
                  <div style={{ position: "relative", background: "#050810" }}>
                    <canvas ref={ref} style={{ display: "block", width: "100%", imageRendering: "pixelated" }} />
                    {processing && badge === "OUTPUT" && (
                      <div style={{ position: "absolute", inset: 0, display: "flex",
                        alignItems: "center", justifyContent: "center",
                        background: "#00000088", fontSize: 13, color: C.accent }}>
                        ⏳ Processing…
                      </div>
                    )}
                    {!loaded && (
                      <div style={{ position: "absolute", inset: 0, display: "flex",
                        alignItems: "center", justifyContent: "center",
                        background: "#050810", fontSize: 12, color: C.muted, flexDirection: "column", gap: 8 }}>
                        <span style={{ fontSize: 32 }}>🩻</span>
                        Click "Generate" to start
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Stats */}
            {stats && (
              <div style={{ background: C.card, borderRadius: 12, border: `1px solid ${C.border}`, padding: 16 }}>
                <div style={{ fontSize: 11, color: C.muted, marginBottom: 10, letterSpacing: 1, fontWeight: 600 }}>
                  📊 IMAGE QUALITY METRICS
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
                  <StatBox label="SNR" value={stats.snr} />
                  <StatBox label="CONTRAST (σ)" value={stats.contrast} />
                  <StatBox label="MEAN INTENSITY" value={stats.mean} />
                </div>
              </div>
            )}

            {/* History */}
            {history.length > 0 && (
              <div style={{ background: C.card, borderRadius: 12, border: `1px solid ${C.border}`,
                padding: "10px 16px", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontSize: 11, color: C.muted, letterSpacing: 1, fontWeight: 600 }}>PIPELINE:</span>
                {history.map((h, i) => (
                  <span key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {i > 0 && <span style={{ color: C.muted }}>→</span>}
                    <Badge label={h.toUpperCase()} color={C.accent} />
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* ── Right: Controls ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Quick Presets */}
            <div style={{ background: C.card, borderRadius: 12, border: `1px solid ${C.border}`, padding: 16 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 12, letterSpacing: 1, fontWeight: 600 }}>
                ⚡ IMAGING PRESETS
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {PRESETS.map(p => (
                  <button key={p.id} onClick={() => applyPreset(p)} style={{
                    padding: "10px 12px", borderRadius: 8, border: `1px solid ${C.border}`,
                    background: C.bg, color: C.text, cursor: "pointer", fontSize: 12,
                    fontWeight: 600, textAlign: "left"
                  }}>
                    <div style={{ fontSize: 18, marginBottom: 4 }}>{p.icon}</div>
                    <div>{p.label}</div>
                    <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>
                      {p.steps.join(" → ")}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Techniques */}
            {Object.entries(groupLabel).map(([grp, grpLabel]) => {
              const techs = TECHNIQUES.filter(t => t.group === grp);
              return (
                <div key={grp} style={{ background: C.card, borderRadius: 12,
                  border: `1px solid ${C.border}`, padding: 16 }}>
                  <div style={{ fontSize: 11, marginBottom: 10, letterSpacing: 1,
                    fontWeight: 700, color: GROUP_COLORS[grp] }}>
                    {grpLabel.toUpperCase()}
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {techs.map(t => (
                      <TechButton key={t.id} {...t} active={active === t.id}
                        onClick={applyTech} />
                    ))}
                  </div>
                </div>
              );
            })}

            {/* Parameters */}
            <div style={{ background: C.card, borderRadius: 12, border: `1px solid ${C.border}`, padding: 16 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 12, letterSpacing: 1, fontWeight: 600 }}>
                ⚙ PARAMETERS
              </div>
              {[
                { key: "clipLimit", label: "CLAHE Clip Limit", min: 0.5, max: 8, step: 0.5 },
                { key: "gamma", label: "Gamma (γ)", min: 0.3, max: 3.0, step: 0.1 },
                { key: "kernelSize", label: "Kernel Size", min: 3, max: 11, step: 2 },
                { key: "strength", label: "Unsharp Strength", min: 0.5, max: 3.0, step: 0.1 },
                { key: "low", label: "Canny Low Threshold", min: 10, max: 100, step: 5 },
                { key: "high", label: "Canny High Threshold", min: 50, max: 250, step: 10 },
              ].map(({ key, label, min, max, step }) => (
                <div key={key} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 11, color: C.muted, marginBottom: 4 }}>
                    <span>{label}</span>
                    <span style={{ color: C.accent, fontFamily: "monospace" }}>{params[key]}</span>
                  </div>
                  <input type="range" min={min} max={max} step={step} value={params[key]}
                    onChange={e => setParams(p => ({ ...p, [key]: +e.target.value }))}
                    style={{ width: "100%", accentColor: C.accent }} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <div style={{ maxWidth: 1200, margin: "24px auto 0", padding: "0 24px" }}>
        <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 16,
          display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <div style={{ fontSize: 12, color: C.muted }}>
            © 2026 <span style={{ color: C.accent, fontWeight: 700 }}>Ali Hussein Allawi</span>
            &nbsp;— All Rights Reserved
          </div>
          <div style={{ fontSize: 11, color: C.muted, textAlign: "right" }}>
            Department of Medical Physics · College of Sciences · University of Al-Qadisiyah, Iraq
          </div>
        </div>
      </div>

    </div>
  );
}
