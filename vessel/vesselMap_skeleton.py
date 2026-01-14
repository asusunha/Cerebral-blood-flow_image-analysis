import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import skeletonize

class FullyAutomatedVesselAnalyzer:

    def assess_image_quality(self, gray):
        """이미지 품질 자동 평가"""
        contrast = gray.std()
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_estimate = np.median(np.abs(laplacian))
        snr = gray.mean() / (noise_estimate + 1e-10)
        
        brightness = gray.mean()
        
        return {
            'contrast': contrast,
            'snr': snr,
            'brightness': brightness,
            'is_low_contrast': contrast < 30,
            'is_noisy': snr < 10,
            'is_dark': brightness < 80,
            'is_bright': brightness > 180
        }

    def estimate_vessel_scale(self, gray):
        """혈관 반경 자동 추정"""
        sigmas = np.arange(1, 16, 2)
        vesselness = frangi(gray, sigmas=sigmas, black_ridges=True)

        mask = vesselness > np.percentile(vesselness, 85)
        if np.count_nonzero(mask) < 100:
            return 5

        radius_map = np.argmax(
            np.stack([frangi(gray, sigmas=[s], black_ridges=True) for s in sigmas]),
            axis=0
        )

        median_radius = np.median(radius_map[mask])
        return max(2, int(median_radius))

    def get_auto_roi(self, raw, vessel_radius):
        """ROI 자동 검출"""
        lab = cv2.cvtColor(raw, cv2.COLOR_BGR2Lab)
        a = lab[:,:,1]

        blur_size = int(6 * vessel_radius) | 1
        blur = cv2.GaussianBlur(a, (blur_size, blur_size), 0)

        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k = int(4 * vessel_radius)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        roi = np.zeros(raw.shape[:2], np.uint8)
        cv2.drawContours(roi, [hull], -1, 255, -1)

        return roi

    def get_adaptive_params(self, quality_metrics, vessel_radius, attempt=0):
        """품질 지표에 따라 파라미터 자동 설정 (더 보수적)"""
        params = {
            'clahe_clip': 2.5,  # 약간 증가
            'vessel_percentile': 88,  # 더 엄격하게 (이전 85)
            'thresh_coef': 0.3,  # 더 보수적 (이전 0.4)
            'morph_scale': 1.2,  # 감소 (이전 1.5)
            'morph_iterations': max(1, vessel_radius // 3)  # 감소
        }
        
        # 대비 낮음
        if quality_metrics['is_low_contrast']:
            params['clahe_clip'] = 3.5
            params['thresh_coef'] = 0.4
            
        # 노이즈 많음
        if quality_metrics['is_noisy']:
            params['vessel_percentile'] = 92  # 더 엄격
            params['morph_scale'] = 1.5
            
        # 어두움
        if quality_metrics['is_dark']:
            params['clahe_clip'] += 0.5
            params['thresh_coef'] = 0.35
            
        # 밝음
        if quality_metrics['is_bright']:
            params['thresh_coef'] = 0.25
            
        # 재시도 시 파라미터 조정
        if attempt == 1:
            params['thresh_coef'] += 0.05
            params['vessel_percentile'] -= 2
        elif attempt == 2:
            params['thresh_coef'] += 0.1
            params['vessel_percentile'] -= 5
            
        return params

    def validate_vessel_extraction(self, vessel_map, roi):
        """혈관 추출 결과 검증 (더 엄격한 기준)"""
        roi_area = np.count_nonzero(roi)
        vessel_area = np.count_nonzero(vessel_map)
        vessel_ratio = vessel_area / roi_area if roi_area > 0 else 0
        
        # 더 좁은 정상 범위
        if vessel_ratio < 0.08:  # 8% 미만 (이전 5%)
            return 'under', vessel_ratio
        elif vessel_ratio > 0.25:  # 25% 초과 (이전 40%)
            return 'over', vessel_ratio
        else:
            return 'ok', vessel_ratio

    def extract_vessels_multiscale(self, enhanced, roi, vessel_radius, params):
        """다중 스케일 혈관 검출 및 통합 (보수적 접근)"""
        
        # 1. 굵은/중간 혈관 먼저 검출 (높은 신뢰도)
        medium_sigmas = range(3, 10, 2)
        medium_frangi = frangi(enhanced, sigmas=medium_sigmas, black_ridges=True)
        
        thick_sigmas = range(8, 16, 3)
        thick_frangi = frangi(enhanced, sigmas=thick_sigmas, black_ridges=True)
        
        # 주요 혈관 마스크 (높은 임계값)
        main_vessels_mask = (medium_frangi > np.percentile(medium_frangi[roi > 0], 88)) | \
                           (thick_frangi > np.percentile(thick_frangi[roi > 0], 90))
        
        # 2. 밝기 기반 필터링 (혈관은 어두워야 함)
        roi_pixels = enhanced[roi > 0]
        intensity_threshold = np.percentile(roi_pixels, 40)  # 하위 40%만
        dark_mask = enhanced < intensity_threshold
        
        # 3. Frangi + 밝기 조건 동시 만족
        vessel_candidates = main_vessels_mask & dark_mask
        
        # 4. 미세혈관은 주요 혈관 근처에서만 검출
        fine_sigmas = range(1, 4, 1)
        fine_frangi = frangi(enhanced, sigmas=fine_sigmas, black_ridges=True)
        
        # 주요 혈관 확장 (근처 영역)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   (vessel_radius * 3, vessel_radius * 3))
        vessel_vicinity = cv2.dilate(vessel_candidates.astype(np.uint8), kernel_dilate)
        
        # 근처에서만 미세혈관 검출
        fine_mask = (fine_frangi > np.percentile(fine_frangi[roi > 0], 92)) & \
                    (vessel_vicinity > 0) & dark_mask
        
        # 5. 통합 (보수적)
        combined = vessel_candidates | fine_mask
        
        return (combined * 255).astype(np.uint8)

    def fill_vessel_holes(self, binary):
        """굵은 혈관 내부 구멍 채우기"""
        # Flood fill from edges to find interior holes
        h, w = binary.shape
        filled = binary.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Invert to find holes
        inverted = cv2.bitwise_not(binary)
        
        # Fill from border (background)
        cv2.floodFill(inverted, mask, (0, 0), 255)
        
        # Invert back - holes are now filled
        inverted = cv2.bitwise_not(inverted)
        
        # Combine with original
        filled = cv2.bitwise_or(binary, inverted)
        
        return filled

    def connect_broken_vessels(self, binary, vessel_radius):
        """끊어진 혈관 연결 및 노이즈 제거 (강화)"""
        
        # 1. 형태학적 Opening으로 작은 노이즈 먼저 제거
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # 2. 작은 구멍 채우기
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # 3. 중간 크기 간격 연결
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   (max(3, vessel_radius // 2), 
                                                    max(3, vessel_radius // 2)))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        # 4. 연결된 성분 분석 (노이즈 제거)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        # 면적 기준으로 노이즈 제거 (더 엄격하게)
        total_area = closed.shape[0] * closed.shape[1]
        min_area = max(50, total_area * 0.0002)  # 최소 50픽셀 또는 0.02%
        
        cleaned = np.zeros_like(closed)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255
        
        # 5. 굵은 혈관 내부 채우기 (선택적)
        filled = self.fill_vessel_holes(cleaned)
        
        # 6. 최종 Smoothing (경계 부드럽게)
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        smoothed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        
        return smoothed

    def extract_vessels_with_params(self, raw, roi, vessel_radius, params):
        """주어진 파라미터로 혈관 추출 (다중 스케일 + 연결성 보완)"""
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=roi)

        tile_size = max(8, vessel_radius * 2)
        clahe = cv2.createCLAHE(
            clipLimit=params['clahe_clip'], 
            tileGridSize=(tile_size, tile_size)
        )
        enhanced = clahe.apply(masked)

        # 다중 스케일 혈관 검출
        multi_scale_binary = self.extract_vessels_multiscale(enhanced, roi, vessel_radius, params)
        
        # 연결성 보완 및 노이즈 제거
        connected = self.connect_broken_vessels(multi_scale_binary, vessel_radius)
        
        # ROI 내부만 유지
        final = cv2.bitwise_and(connected, connected, mask=roi)
        
        return final

    def get_adaptive_vessels(self, raw, roi, vessel_radius, quality_metrics):
        """자동 재시도 메커니즘을 포함한 혈관 추출"""
        max_attempts = 3
        best_vessel = None
        best_params = None
        best_score = float('inf')
        
        for attempt in range(max_attempts):
            params = self.get_adaptive_params(quality_metrics, vessel_radius, attempt)
            vessel = self.extract_vessels_with_params(raw, roi, vessel_radius, params)
            
            status, ratio = self.validate_vessel_extraction(vessel, roi)
            
            # 최적 비율(15%)과의 거리로 점수 계산
            score = abs(ratio - 0.15)
            
            if score < best_score:
                best_score = score
                best_vessel = vessel
                best_params = params
            
            if status == 'ok':
                print(f"✓ 시도 {attempt+1}: 성공 (혈관 비율: {ratio:.1%})")
                return vessel, params, True
            else:
                print(f"⚠ 시도 {attempt+1}: {status} (혈관 비율: {ratio:.1%})")
        
        print(f"→ 최선의 결과 사용 (혈관 비율: {best_score+0.15:.1%})")
        return best_vessel, best_params, False

    def analyze(self, img_path):
        """전체 파이프라인"""
        raw = cv2.imread(img_path)
        if raw is None:
            print("❌ 이미지 로드 실패")
            return

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        
        # 1. 이미지 품질 평가
        quality = self.assess_image_quality(gray)
        print(f"\n📊 이미지 품질 분석:")
        print(f"  - 대비: {quality['contrast']:.1f} {'(낮음)' if quality['is_low_contrast'] else '(양호)'}")
        print(f"  - SNR: {quality['snr']:.1f} {'(노이즈 많음)' if quality['is_noisy'] else '(양호)'}")
        print(f"  - 밝기: {quality['brightness']:.1f}")

        # 2. 혈관 반경 추정
        vessel_radius = self.estimate_vessel_scale(gray)
        print(f"\n🔍 자동 추정 혈관 반경: {vessel_radius}px")

        # 3. ROI 검출
        roi = self.get_auto_roi(raw, vessel_radius)
        if roi is None:
            print("❌ ROI 검출 실패")
            return

        # 4. 적응적 혈관 추출
        print(f"\n🎯 혈관 추출 시도:")
        vessel, params, success = self.get_adaptive_vessels(raw, roi, vessel_radius, quality)
        
        print(f"\n⚙️ 최종 사용 파라미터:")
        for k, v in params.items():
            print(f"  - {k}: {v}")

        # 5. 골격화
        skeleton = skeletonize(vessel > 0).astype(np.uint8) * 255

        # 6. 시각화
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(vessel, cmap='gray')
        plt.title(f"Vessel Map {'✓' if success else '⚠'}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeleton")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyzer = FullyAutomatedVesselAnalyzer()
    img_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Baseline_ROI\base_f00_original.tiff"
    analyzer.analyze(img_path)