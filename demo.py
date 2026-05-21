"""
demo.py – عرض توضيحي كامل لنظام تحسين الصور الطبية
يولّد صورة اصطناعية تشبه الأشعة السينية ثم يطبق جميع التقنيات
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# إضافة المجلد الحالي إلى مسار Python
sys.path.insert(0, str(Path(__file__).parent))
from medical_processor import MedicalImageEnhancer


def create_synthetic_xray(size: int = 512) -> np.ndarray:
    """
    إنشاء صورة اصطناعية تحاكي الأشعة السينية للرئة
    تحتوي على: ضوضاء، تباين منخفض، هياكل دائرية
    """
    rng = np.random.default_rng(42)
    img = np.zeros((size, size), dtype=np.float32)

    # خلفية رمادية مع تدرج
    cx, cy = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            img[y, x] = max(0, 1 - dist / (size * 0.6))

    # إضافة هياكل دائرية (محاكاة العقد)
    for _ in range(8):
        rx = rng.integers(size // 4, 3 * size // 4)
        ry = rng.integers(size // 4, 3 * size // 4)
        r = rng.integers(15, 50)
        intensity = rng.uniform(0.2, 0.7)
        y_grid, x_grid = np.ogrid[:size, :size]
        mask = (x_grid - rx)**2 + (y_grid - ry)**2 <= r**2
        img[mask] = np.clip(img[mask] + intensity, 0, 1)

    # إضافة ضوضاء غاوسية
    noise = rng.normal(0, 0.08, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)

    return (img * 255).astype(np.uint8)


def print_stats(name: str, original: np.ndarray, processed: np.ndarray):
    """طباعة إحصائيات المقارنة"""
    enhancer = MedicalImageEnhancer()
    snr = enhancer.calculate_snr(processed)
    contrast = enhancer.calculate_contrast(processed)
    psnr = enhancer.calculate_psnr(original, processed)
    print(f"  {'─'*40}")
    print(f"  📊 {name}")
    print(f"     SNR     : {snr:.2f}")
    print(f"     Contrast: {contrast:.2f}")
    print(f"     PSNR    : {psnr:.2f} dB")


def run_demo():
    print("\n" + "="*55)
    print("  🏥  نظام تحسين الصور الطبية الإشعاعية")
    print("  Medical Radiological Image Enhancement System")
    print("="*55 + "\n")

    # إعداد المجلدات
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # إنشاء المعالج والصورة
    enhancer = MedicalImageEnhancer()
    print("✅  توليد صورة أشعة سينية اصطناعية...")
    xray = create_synthetic_xray(512)
    cv2.imwrite(str(output_dir / "00_original.png"), xray)
    enhancer.original = xray.copy()

    # ── 1. تحسين التباين ──────────────────────────────
    print("\n📌  [1] تحسين التباين (Contrast Enhancement)")
    clahe = enhancer.clahe_enhancement(xray, clip_limit=2.5)
    hist_eq = enhancer.histogram_equalization(xray)
    gamma = enhancer.gamma_correction(xray, gamma=1.4)

    cv2.imwrite(str(output_dir / "01_clahe.png"), clahe)
    cv2.imwrite(str(output_dir / "02_hist_eq.png"), hist_eq)
    cv2.imwrite(str(output_dir / "03_gamma.png"), gamma)

    print_stats("CLAHE", xray, clahe)
    print_stats("Histogram Equalization", xray, hist_eq)
    print_stats("Gamma Correction (γ=1.4)", xray, gamma)

    # ── 2. تقليل الضوضاء ─────────────────────────────
    print("\n📌  [2] تقليل الضوضاء (Noise Reduction)")
    gaussian = enhancer.gaussian_blur(xray, kernel_size=5, sigma=1.0)
    median = enhancer.median_filter(xray, kernel_size=5)
    bilateral = enhancer.bilateral_filter(xray)
    nlm = enhancer.non_local_means(xray, h=10)

    cv2.imwrite(str(output_dir / "04_gaussian.png"), gaussian)
    cv2.imwrite(str(output_dir / "05_median.png"), median)
    cv2.imwrite(str(output_dir / "06_bilateral.png"), bilateral)
    cv2.imwrite(str(output_dir / "07_nlm.png"), nlm)

    print_stats("Gaussian Blur", xray, gaussian)
    print_stats("Median Filter", xray, median)
    print_stats("Bilateral Filter", xray, bilateral)
    print_stats("Non-Local Means", xray, nlm)

    # ── 3. كشف الحواف ────────────────────────────────
    print("\n📌  [3] كشف الحواف (Edge Detection)")
    canny = enhancer.canny_edges(clahe, 30, 120)
    sobel = enhancer.sobel_edges(clahe)
    laplacian = enhancer.laplacian_edges(clahe)
    unsharp = enhancer.unsharp_masking(bilateral, strength=1.5)

    cv2.imwrite(str(output_dir / "08_canny.png"), canny)
    cv2.imwrite(str(output_dir / "09_sobel.png"), sobel)
    cv2.imwrite(str(output_dir / "10_laplacian.png"), laplacian)
    cv2.imwrite(str(output_dir / "11_unsharp.png"), unsharp)

    print_stats("Canny Edges", xray, canny)
    print_stats("Sobel Edges", xray, sobel)
    print_stats("Laplacian", xray, laplacian)
    print_stats("Unsharp Masking", xray, unsharp)

    # ── 4. Pipeline متكامل ────────────────────────────
    print("\n📌  [4] خطوط المعالجة الكاملة (Full Pipelines)")
    for preset in ["xray", "mri", "ct", "mammo"]:
        results = enhancer.full_enhancement_pipeline(xray, preset=preset)
        for step, img in results.items():
            if step != "original":
                cv2.imwrite(str(output_dir / f"pipeline_{preset}_{step}.png"), img)
        print(f"  ✔  {preset.upper()} pipeline → {len(results)-1} مخرجات")

    # ── 5. إنشاء مقارنة بصرية ────────────────────────
    print("\n📌  [5] إنشاء صورة مقارنة شاملة...")
    _create_comparison_grid(xray, enhancer, output_dir)

    # ── 6. تقرير JSON ─────────────────────────────────
    print("\n📌  [6] حفظ تقرير التحليل...")
    pipeline_results = enhancer.full_enhancement_pipeline(xray, preset="xray")
    report = enhancer.generate_report(
        pipeline_results,
        output_path=str(output_dir / "report.json")
    )

    print("\n" + "="*55)
    print("  ✅  اكتمل العرض التوضيحي بنجاح!")
    print(f"  📁  المخرجات محفوظة في: {output_dir.resolve()}/")
    print("="*55 + "\n")


def _create_comparison_grid(original, enhancer, output_dir):
    """إنشاء شبكة مقارنة 4×3 لأبرز التقنيات"""
    imgs = {
        "Original": original,
        "CLAHE": enhancer.clahe_enhancement(original),
        "Histogram Eq.": enhancer.histogram_equalization(original),
        "Gamma γ=1.4": enhancer.gamma_correction(original, 1.4),
        "Gaussian Blur": enhancer.gaussian_blur(original),
        "Median Filter": enhancer.median_filter(original),
        "Bilateral": enhancer.bilateral_filter(original),
        "Non-Local Means": enhancer.non_local_means(original),
        "Canny Edges": enhancer.canny_edges(original),
        "Sobel Edges": enhancer.sobel_edges(original),
        "Laplacian": enhancer.laplacian_edges(original),
        "Unsharp Mask": enhancer.unsharp_masking(original),
    }

    cols, rows = 4, 3
    thumb = 200
    pad = 30
    header = 35
    W = cols * thumb + (cols + 1) * pad
    H = rows * thumb + rows * header + (rows + 1) * pad

    grid = np.ones((H, W), dtype=np.uint8) * 20  # خلفية داكنة

    for idx, (title, img) in enumerate(imgs.items()):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb + pad)
        y = pad + row * (thumb + header + pad)

        resized = cv2.resize(img, (thumb, thumb))
        grid[y + header: y + header + thumb, x: x + thumb] = resized

        # عنوان مصغّر بالخط الافتراضي
        cv2.putText(
            grid, title,
            (x, y + header - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            220, 1, cv2.LINE_AA
        )

    cv2.imwrite(str(output_dir / "comparison_grid.png"), grid)
    print("  ✔  comparison_grid.png تم الإنشاء")


if __name__ == "__main__":
    run_demo()
