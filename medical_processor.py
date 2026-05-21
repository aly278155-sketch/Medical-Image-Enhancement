"""
================================================================
   Medical Image Enhancement System
   For Radiological Image Processing

   Developer  : Ali Hussein Allawi
   Department : Medical Physics
   College    : College of Sciences
   University : University of Al-Qadisiyah, Iraq
   Year       : 2026

   All Rights Reserved © 2026
================================================================
"""
Medical Image Enhancement System
==================================
نظام تحسين الصور الطبية باستخدام OpenCV
يدعم: تحسين التباين، تقليل الضوضاء، كشف الحواف
"""

import cv2
import numpy as np
from pathlib import Path
import json
import base64
from typing import Optional


class MedicalImageEnhancer:
    """معالج الصور الطبية - يطبق تقنيات متعددة لتحسين الجودة"""

    def __init__(self):
        self.original = None
        self.processed = None
        self.history = []

    # ───────────────────────────────────────────────
    # تحميل الصورة
    # ───────────────────────────────────────────────
    def load_image(self, path: str) -> np.ndarray:
        """تحميل صورة من ملف"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"تعذّر تحميل الصورة: {path}")
        self.original = img.copy()
        self.processed = img.copy()
        return img

    def load_from_bytes(self, data: bytes) -> np.ndarray:
        """تحميل صورة من بيانات ثنائية (bytes)"""
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("تنسيق الصورة غير مدعوم")
        self.original = img.copy()
        self.processed = img.copy()
        return img

    # ───────────────────────────────────────────────
    # 1. تحسين التباين (Contrast Enhancement)
    # ───────────────────────────────────────────────
    def clahe_enhancement(
        self,
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: int = 8
    ) -> np.ndarray:
        """
        CLAHE – Contrast Limited Adaptive Histogram Equalization
        مثالي للأشعة السينية والرنين المغناطيسي
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        return clahe.apply(img)

    def histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """تسوية الهيستوغرام – تحسين التباين العالمي"""
        return cv2.equalizeHist(img)

    def gamma_correction(
        self, img: np.ndarray, gamma: float = 1.5
    ) -> np.ndarray:
        """تصحيح جاما – ضبط السطوع بشكل غير خطي"""
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8
        )
        return cv2.LUT(img, table)

    def adaptive_histogram_eq(self, img: np.ndarray) -> np.ndarray:
        """تسوية هيستوغرام تكيّفية محلية"""
        return self.clahe_enhancement(img, clip_limit=3.0, tile_size=16)

    # ───────────────────────────────────────────────
    # 2. تقليل الضوضاء (Noise Reduction)
    # ───────────────────────────────────────────────
    def gaussian_blur(
        self, img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0
    ) -> np.ndarray:
        """تمرير منخفض بفلتر غاوسي"""
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(img, (k, k), sigma)

    def median_filter(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """فلتر الوسيط – ممتاز لإزالة ضوضاء salt-and-pepper"""
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(img, k)

    def bilateral_filter(
        self,
        img: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """فلتر ثنائي – يزيل الضوضاء مع الحفاظ على الحواف"""
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def non_local_means(
        self,
        img: np.ndarray,
        h: float = 10,
        template_window: int = 7,
        search_window: int = 21
    ) -> np.ndarray:
        """Non-Local Means – أقوى طريقة لإزالة الضوضاء"""
        return cv2.fastNlMeansDenoising(
            img, None, h, template_window, search_window
        )

    # ───────────────────────────────────────────────
    # 3. كشف الحواف (Edge Detection)
    # ───────────────────────────────────────────────
    def canny_edges(
        self,
        img: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """Canny – أشهر خوارزمية لكشف الحواف"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def sobel_edges(
        self, img: np.ndarray, ksize: int = 3
    ) -> np.ndarray:
        """Sobel – كشف الحواف بالمشتقات الأولى"""
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def laplacian_edges(self, img: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Laplacian – المشتقة الثانية، حساسة للتفاصيل الدقيقة"""
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
        return cv2.convertScaleAbs(lap)

    def unsharp_masking(
        self, img: np.ndarray, strength: float = 1.5, blur_size: int = 5
    ) -> np.ndarray:
        """Unsharp Masking – تحسين الحدة بتقنية القناع الغير حاد"""
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ───────────────────────────────────────────────
    # معالجات مركّبة (Pipelines)
    # ───────────────────────────────────────────────
    def full_enhancement_pipeline(
        self, img: np.ndarray, preset: str = "xray"
    ) -> dict:
        """
        خطوط معالجة جاهزة حسب نوع الصورة:
        - xray  : أشعة سينية
        - mri   : رنين مغناطيسي
        - ct    : أشعة مقطعية
        - mammo : تصوير الثدي بالأشعة
        """
        results = {"original": img}

        if preset == "xray":
            denoised = self.bilateral_filter(img)
            enhanced = self.clahe_enhancement(denoised, clip_limit=2.5)
            edges = self.canny_edges(enhanced)
            results.update({"denoised": denoised, "enhanced": enhanced, "edges": edges})

        elif preset == "mri":
            denoised = self.non_local_means(img, h=8)
            enhanced = self.clahe_enhancement(denoised, clip_limit=3.0, tile_size=16)
            edges = self.unsharp_masking(enhanced, strength=1.2)
            results.update({"denoised": denoised, "enhanced": enhanced, "edges": edges})

        elif preset == "ct":
            denoised = self.gaussian_blur(img, kernel_size=3, sigma=0.8)
            enhanced = self.gamma_correction(denoised, gamma=1.3)
            edges = self.sobel_edges(enhanced)
            results.update({"denoised": denoised, "enhanced": enhanced, "edges": edges})

        elif preset == "mammo":
            denoised = self.median_filter(img, kernel_size=3)
            enhanced = self.clahe_enhancement(denoised, clip_limit=4.0, tile_size=8)
            edges = self.laplacian_edges(enhanced)
            results.update({"denoised": denoised, "enhanced": enhanced, "edges": edges})

        return results

    # ───────────────────────────────────────────────
    # مقاييس الجودة (Quality Metrics)
    # ───────────────────────────────────────────────
    def calculate_snr(self, img: np.ndarray) -> float:
        """نسبة الإشارة إلى الضوضاء (SNR)"""
        mean = np.mean(img)
        std = np.std(img)
        return float(mean / std) if std > 0 else 0.0

    def calculate_contrast(self, img: np.ndarray) -> float:
        """التباين الجذري المتوسط المربع (RMS Contrast)"""
        return float(np.std(img))

    def calculate_psnr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """نسبة الإشارة إلى الضوضاء الأقصى (PSNR)"""
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    def histogram_stats(self, img: np.ndarray) -> dict:
        """إحصائيات الهيستوغرام"""
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        return {
            "mean": float(np.mean(img)),
            "std": float(np.std(img)),
            "min": int(img.min()),
            "max": int(img.max()),
            "histogram": hist.tolist()
        }

    # ───────────────────────────────────────────────
    # تصدير النتائج
    # ───────────────────────────────────────────────
    def save_image(self, img: np.ndarray, path: str) -> bool:
        """حفظ الصورة إلى ملف"""
        return cv2.imwrite(path, img)

    def to_base64(self, img: np.ndarray) -> str:
        """تحويل الصورة إلى base64 للعرض في المتصفح"""
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_report(
        self, results: dict, output_path: Optional[str] = None
    ) -> dict:
        """توليد تقرير تحليلي شامل"""
        report = {}
        for name, img in results.items():
            report[name] = {
                "shape": img.shape,
                "dtype": str(img.dtype),
                "snr": self.calculate_snr(img),
                "contrast": self.calculate_contrast(img),
                **self.histogram_stats(img)
            }
            if name != "original" and "original" in results:
                report[name]["psnr"] = self.calculate_psnr(results["original"], img)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                serializable = {
                    k: {kk: vv for kk, vv in v.items() if kk != "histogram"}
                    for k, v in report.items()
                }
                json.dump(serializable, f, ensure_ascii=False, indent=2)

        return report
