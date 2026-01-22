import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
import os

class IntegratedVesselSystem:
    def __init__(self, 
                 # 1번 코드 파라미터 (문서 4)
                 scale_groups_code1=None, 
                 threshold_val_code1=3.0, 
                 min_size_code1=300, 
                 second_threshold_code1=110, 
                 smoothing_sigma_code1=1.5,
                 # 2번 코드 파라미터 (문서 5)
                 scale_groups_code2=None,
                 threshold_val_code2=3.0,
                 min_size_code2=300,
                 second_threshold_code2=110,
                 smoothing_sigma_code2=1.5):
        """
        통합 혈관 검출 시스템
        - 1번 코드(문서 4): 기본 Sato 필터 기반
        - 2번 코드(문서 5): 개선된 적응형 임계값 기반
        - 1번 코드의 배경 영역에서 2번 코드 결과 제거
        """
        # 1번 코드 파라미터
        if scale_groups_code1 is None:
            scale_groups_code1 = [
                range(1, 3, 1), 
                range(3, 6, 1), 
                range(6, 10, 1)
            ]
        self.scale_groups_code1 = scale_groups_code1
        self.threshold_val_code1 = threshold_val_code1
        self.min_size_code1 = min_size_code1
        self.second_threshold_code1 = second_threshold_code1
        self.smoothing_sigma_code1 = smoothing_sigma_code1
        
        # 2번 코드 파라미터
        if scale_groups_code2 is None:
            scale_groups_code2 = [
                range(1, 3, 1), 
                range(3, 6, 1), 
                range(6, 10, 1)
            ]
        self.scale_groups_code2 = scale_groups_code2
        self.threshold_val_code2 = threshold_val_code2
        self.min_size_code2 = min_size_code2
        self.second_threshold_code2 = second_threshold_code2
        self.smoothing_sigma_code2 = smoothing_sigma_code2

    # ========== 1번 코드 메서드 (문서 4 - 기본 버전) ==========
    def get_first_stage_vessels_code1(self, raw_img, roi_mask):
        green_channel = raw_img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        inverted = cv2.bitwise_not(enhanced)
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        
        max_prob = np.zeros(raw_img.shape[:2], dtype=np.float32)
        for sigmas in self.scale_groups_code1:
            vessel_prob = sato(blurred, sigmas=sigmas, black_ridges=False)
            max_prob = np.maximum(max_prob, vessel_prob)
        
        vessel = (max_prob > self.threshold_val_code1).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, kernel)
        
        binary_bool = vessel > 0
        cleaned_bool = remove_small_objects(binary_bool, min_size=self.min_size_code1)
        vessel = cleaned_bool.astype(np.uint8) * 255
        
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask), enhanced

    def get_second_stage_vessels_code1(self, raw_img, first_stage_mask):
        masked_img = raw_img.copy()
        masked_img[first_stage_mask > 0] = [0, 0, 0]
        
        lab_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
        a_channel = lab_img[:, :, 1]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        a_enhanced = clahe.apply(a_channel)
        
        _, second_vessel = cv2.threshold(a_enhanced, self.second_threshold_code1, 255, cv2.THRESH_BINARY)
        second_vessel = cv2.bitwise_and(second_vessel, cv2.bitwise_not(first_stage_mask))
        
        return second_vessel, a_enhanced

    # ========== 2번 코드 메서드 (문서 5 - 개선 버전) ==========
    def get_first_stage_vessels_code2(self, raw_img, roi_mask):
        green_channel = raw_img[:, :, 1]
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        inverted = cv2.bitwise_not(enhanced)
        
        max_prob = np.zeros(raw_img.shape[:2], dtype=np.float32)
        fine_sigmas = [0.5, 1.0, 1.5]
        all_sigmas = fine_sigmas + [s for g in self.scale_groups_code2 for s in g]
        
        for s in all_sigmas:
            vessel_prob = sato(inverted, sigmas=(s,), black_ridges=False)
            max_prob = np.maximum(max_prob, vessel_prob)

        # 적응형 임계값 처리
        max_prob_uint8 = cv2.normalize(max_prob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        vessel = cv2.adaptiveThreshold(
            max_prob_uint8, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, kernel)
        
        binary_bool = vessel > 0
        cleaned_bool = remove_small_objects(binary_bool, min_size=100)
        vessel = cleaned_bool.astype(np.uint8) * 255
        
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask), enhanced

    def get_second_stage_vessels_code2(self, raw_img, first_stage_mask):
        lab_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB)
        a_channel = lab_img[:, :, 1]
        
        clahe_a = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
        a_enhanced = clahe_a.apply(a_channel)
        
        if self.second_threshold_code2 > 0:
            ret, a_thresh = cv2.threshold(a_enhanced, self.second_threshold_code2, 255, cv2.THRESH_BINARY)
            self.computed_otsu_val = self.second_threshold_code2
        else:
            ret, a_thresh = cv2.threshold(a_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.computed_otsu_val = ret
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_seed = cv2.dilate(first_stage_mask, kernel, iterations=1)
        
        second_vessel = cv2.bitwise_or(
            cv2.bitwise_and(a_thresh, dilated_seed),
            cv2.bitwise_and(a_thresh, cv2.bitwise_not(first_stage_mask))
        )
        
        return second_vessel, a_enhanced

    # ========== 공통 메서드 ==========
    def create_vessel_overlay(self, raw_img, vessel_mask):
        """혈관 마스크를 원본 이미지에 초록색으로 오버레이"""
        overlay = raw_img.copy()
        mask_indices = vessel_mask > 0
        if np.any(mask_indices):
            overlay[mask_indices] = cv2.addWeighted(
                raw_img[mask_indices], 0.4, 
                np.full_like(raw_img[mask_indices], (0, 255, 0)), 0.6, 0
            )
        return overlay

    # ========== 핵심: 배경 제거 로직 ==========
    def remove_background_vessels(self, code1_vessel_mask, code2_vessel_mask):
        """
        1번 코드의 배경 영역(혈관이 아닌 부분)에서 
        2번 코드의 혈관을 제거
        
        Parameters:
        - code1_vessel_mask: 1번 코드의 최종 혈관 마스크 (binary)
        - code2_vessel_mask: 2번 코드의 최종 혈관 마스크 (binary)
        
        Returns:
        - refined_code2_mask: 배경 영역이 제거된 2번 코드 결과
        """
        # 2번 코드 결과에서 1번 코드의 배경과 겹치는 부분 제거
        # 즉, 2번 결과 AND 1번 혈관 = 1번 혈관 영역에서만 2번 혈관 유지
        refined_code2_mask = cv2.bitwise_and(code2_vessel_mask, code1_vessel_mask)
        
        return refined_code2_mask

    def analyze(self, img_path, save_overlay=True):
        raw = cv2.imread(img_path)
        if raw is None: 
            print(f"이미지를 불러올 수 없습니다: {img_path}")
            return
        
        roi_mask = np.ones(raw.shape[:2], dtype=np.uint8) * 255
        
        print("=" * 60)
        print("1번 코드 실행 중 (기본 Sato 필터)...")
        print("=" * 60)
        # ========== 1번 코드 실행 ==========
        first_stage_c1, green_clahe_c1 = self.get_first_stage_vessels_code1(raw, roi_mask)
        second_stage_c1, a_enhanced_c1 = self.get_second_stage_vessels_code1(raw, first_stage_c1)
        code1_vessel = cv2.bitwise_or(first_stage_c1, second_stage_c1)
        
        print(f"✓ Code1 1차 혈관 픽셀 수: {np.sum(first_stage_c1 > 0):,}")
        print(f"✓ Code1 2차 혈관 픽셀 수: {np.sum(second_stage_c1 > 0):,}")
        print(f"✓ Code1 최종 혈관 픽셀 수: {np.sum(code1_vessel > 0):,}")
        
        print("\n" + "=" * 60)
        print("2번 코드 실행 중 (개선된 적응형 임계값)...")
        print("=" * 60)
        # ========== 2번 코드 실행 ==========
        first_stage_c2, green_clahe_c2 = self.get_first_stage_vessels_code2(raw, roi_mask)
        second_stage_c2, a_enhanced_c2 = self.get_second_stage_vessels_code2(raw, first_stage_c2)
        code2_vessel_original = cv2.bitwise_or(first_stage_c2, second_stage_c2)
        
        print(f"✓ Code2 1차 혈관 픽셀 수: {np.sum(first_stage_c2 > 0):,}")
        print(f"✓ Code2 2차 혈관 픽셀 수: {np.sum(second_stage_c2 > 0):,}")
        print(f"✓ Code2 원본 혈관 픽셀 수: {np.sum(code2_vessel_original > 0):,}")
        
        print("\n" + "=" * 60)
        print("배경 제거 로직 적용 중...")
        print("=" * 60)
        # ========== 배경 제거 로직 적용 ==========
        code2_vessel_refined = self.remove_background_vessels(code1_vessel, code2_vessel_original)
        
        removed_pixels = np.sum(code2_vessel_original > 0) - np.sum(code2_vessel_refined > 0)
        print(f"✓ Code2 정제 후 혈관 픽셀 수: {np.sum(code2_vessel_refined > 0):,}")
        print(f"✓ 제거된 배경 픽셀 수: {removed_pixels:,}")
        print("=" * 60 + "\n")
        
        # ========== 시각화 ==========
        code1_overlay = self.create_vessel_overlay(raw, code1_vessel)
        code2_original_overlay = self.create_vessel_overlay(raw, code2_vessel_original)
        code2_refined_overlay = self.create_vessel_overlay(raw, code2_vessel_refined)
        
        # ========== 지표 계산 (정제된 2번 코드 기준) ==========
        if np.sum(code2_vessel_refined) > 0:
            skeleton = skeletonize(code2_vessel_refined > 0).astype(np.uint8) * 255
        else:
            skeleton = np.zeros_like(code2_vessel_refined)

        dist_map = distance_transform_edt(code2_vessel_refined > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(code2_vessel_refined > 0) / (raw.shape[0] * raw.shape[1])
        
        # ========== 결과 저장 ==========
        if save_overlay:
            base_dir = os.path.dirname(img_path)
            output_dir = os.path.join(base_dir, "output_integrated")
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 1번 코드 결과
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_code1_overlay.png"), code1_overlay)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_code1_binary.png"), code1_vessel)
            
            # 2번 코드 원본 결과
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_code2_original_overlay.png"), code2_original_overlay)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_code2_original_binary.png"), code2_vessel_original)
            
            # 2번 코드 정제 결과
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_code2_refined_overlay.png"), code2_refined_overlay)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_code2_refined_binary.png"), code2_vessel_refined)
            
            print(f"모든 결과 저장 완료: {output_dir}\n")
        
        # ========== 플롯 출력 ==========
        plt.figure(figsize=(20, 12))
        
        imgs = [
            cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
            green_clahe_c1,
            cv2.cvtColor(code1_overlay, cv2.COLOR_BGR2RGB),
            code1_vessel,
            green_clahe_c2,
            cv2.cvtColor(code2_original_overlay, cv2.COLOR_BGR2RGB),
            code2_vessel_original,
            cv2.cvtColor(code2_refined_overlay, cv2.COLOR_BGR2RGB),
            code2_vessel_refined
        ]
        
        titles = [
            '1. Original Image',
            '2. Code1 Green Channel',
            '3. Code1 Result (Overlay)',
            '4. Code1 Binary Mask',
            '5. Code2 Green Channel',
            '6. Code2 Original (Overlay)',
            '7. Code2 Original Binary',
            '8. Code2 Refined (Overlay)',
            '9. Code2 Refined Binary'
        ]
        
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(3, 3, i+1)
            if len(img.shape) == 3:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap='gray')
            plt.title(title, fontsize=10)
            plt.axis('off')
        
        report = (f"Refined Code2 - Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px\n"
                 f"Code1 (기본): 1st Thr={self.threshold_val_code1}, 2nd Thr={self.second_threshold_code1}\n"
                 f"Code2 (개선): Adaptive Threshold + Fine Scales\n"
                 f"배경 제거: Code1 배경 영역에서 Code2 혈관 {removed_pixels:,}px 제거됨")
        plt.figtext(0.5, 0.02, report, ha="center", fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":10})
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    target_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff"
    
    analyzer = IntegratedVesselSystem(
        # 1번 코드 파라미터 (문서 4 - 기본 버전)
        scale_groups_code1=[range(1, 3, 1), range(3, 6, 1), range(6, 12, 1)],
        threshold_val_code1=4.0,
        min_size_code1=400,
        second_threshold_code1=160,
        smoothing_sigma_code1=3.0,
        # 2번 코드 파라미터 (문서 5 - 개선 버전)
        scale_groups_code2=[range(1, 4, 1)],
        threshold_val_code2=2.5,
        min_size_code2=100,
        second_threshold_code2=150,
        smoothing_sigma_code2=0.5
    )
    
    if os.path.exists(target_path):
        analyzer.analyze(target_path)
    else:
        print("파일 경로를 확인해주세요.")