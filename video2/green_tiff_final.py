import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
import os

class ThreeStageVesselSystem:
    def __init__(self, scale_groups=None, threshold_val=3.0, min_size=300, second_threshold=110, smoothing_sigma=1.5):
        """
        scale_groups: 각 스케일 그룹 리스트
        threshold_val: 1차 MAX 확률 맵에 적용할 단일 임계값
        min_size: 제거할 작은 객체의 최소 크기
        second_threshold: 2차 a channel 임계값
        smoothing_sigma: 3차 경계면 평활화 강도 (가우시안 블러 표준편차)
        """
        if scale_groups is None:
            # 기본값도 0.3 ~ 20 범위를 커버하도록 수정
            scale_groups = [
                np.arange(0.3, 3.0, 0.5), 
                np.arange(3.0, 10.0, 1.0), 
                np.arange(10.0, 21.0, 2.0)
            ]
        
        self.scale_groups = scale_groups
        self.threshold_val = threshold_val
        self.min_size = min_size
        self.second_threshold = second_threshold
        self.smoothing_sigma = smoothing_sigma

    def get_first_stage_vessels(self, raw_img, roi_mask):
        green_channel = raw_img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        inverted = cv2.bitwise_not(enhanced)
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        
        max_prob = np.zeros(raw_img.shape[:2], dtype=np.float32)
        for sigmas in self.scale_groups:
            # sato 함수는 numpy array 형태의 sigmas를 정상적으로 인식합니다.
            vessel_prob = sato(blurred, sigmas=sigmas, black_ridges=False)
            max_prob = np.maximum(max_prob, vessel_prob)
        
        vessel = (max_prob > self.threshold_val).astype(np.uint8) * 255
        
        # 원본 모폴로지 로직 유지
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, kernel)
        
        binary_bool = vessel > 0
        cleaned_bool = remove_small_objects(binary_bool, min_size=self.min_size)
        vessel = cleaned_bool.astype(np.uint8) * 255
        
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask), enhanced

    def get_second_stage_vessels(self, raw_img, first_stage_mask):
        masked_img = raw_img.copy()
        masked_img[first_stage_mask > 0] = [0, 0, 0]
        
        lab_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
        a_channel = lab_img[:, :, 1]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        a_enhanced = clahe.apply(a_channel)
        
        _, second_vessel = cv2.threshold(a_enhanced, self.second_threshold, 255, cv2.THRESH_BINARY)
        second_vessel = cv2.bitwise_and(second_vessel, cv2.bitwise_not(first_stage_mask))
        
        return second_vessel, a_enhanced

    def get_third_stage_refinement(self, combined_mask):
        """3차 정제: 모폴로지 없이 가우시안 평활화로 지글거림 제거"""
        if np.sum(combined_mask) == 0:
            return combined_mask
            
        # 1. 미세 노이즈 제거 (boolean 변환 후 작은 객체 삭제)
        cleaned_bool = remove_small_objects(combined_mask > 0, min_size=self.min_size)
        cleaned = cleaned_bool.astype(np.float32) * 255
        
        # 2. 가우시안 평활화 (지글지글한 계단 현상을 부드러운 그라데이션으로 변환)
        refined = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=self.smoothing_sigma)
        
        # 3. 재-이진화 (0.5 수준인 127에서 컷하여 매끄러운 경계 확정)
        _, final_mask = cv2.threshold(refined.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        return final_mask

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

    def analyze(self, img_path, save_overlay=True):
        raw = cv2.imread(img_path)
        if raw is None: 
            print(f"이미지를 불러올 수 없습니다: {img_path}")
            return
        
        roi_mask = np.ones(raw.shape[:2], dtype=np.uint8) * 255
        
        # 1차 혈관 추출
        first_stage, green_clahe = self.get_first_stage_vessels(raw, roi_mask)
        first_stage_overlay = self.create_vessel_overlay(raw, first_stage)
        
        # 2차 혈관 추출
        second_stage, a_enhanced = self.get_second_stage_vessels(raw, first_stage)
        second_stage_overlay = self.create_vessel_overlay(raw, second_stage)
        
        # 임시 합본
        combined = cv2.bitwise_or(first_stage, second_stage)
        
        # 3차 정제 (매끄럽게 가공)
        final_vessel = self.get_third_stage_refinement(combined)
        
        # 최종 결과 시각화
        final_viz = self.create_vessel_overlay(raw, final_vessel)

        # 결과 저장 (최종 결과만)
        if save_overlay:
            base_dir = os.path.dirname(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_vessel_final.png")
            cv2.imwrite(output_path, final_viz)
            print(f"결과 저장 완료: {output_path}")

        # 지표 계산
        if np.sum(final_vessel) > 0:
            skeleton = skeletonize(final_vessel > 0).astype(np.uint8) * 255
        else:
            skeleton = np.zeros_like(final_vessel)

        dist_map = distance_transform_edt(final_vessel > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(final_vessel > 0) / (raw.shape[0] * raw.shape[1])
        
        # 플롯 출력
        imgs = [raw, green_clahe, first_stage_overlay, a_enhanced, second_stage_overlay, final_viz]
        titles = ['Original', '1st Channel\n(Green+CLAHE)', '1st Result\n(Overlay)', 
                  '2nd Channel\n(a+CLAHE)', '2nd Result\n(Overlay)', 'Final Result']
        
        plt.figure(figsize=(18, 10))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(2, 3, i+1)
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        report = (f"Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px\n"
                  f"1st Thr: {self.threshold_val} | 2nd Thr: {self.second_threshold} | Smooth Sigma: {self.smoothing_sigma}")
        plt.figtext(0.5, 0.02, report, ha="center", fontsize=12, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":10})
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    target_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff"
    
    # 0.3부터 20까지 범위를 np.arange로 수정
    analyzer = ThreeStageVesselSystem(
        scale_groups=[
            np.arange(0.3, 3.0, 0.5),   # 미세 혈관 (0.3 ~ 2.8)
            np.arange(3.0, 10.0, 1.0),  # 중간 혈관 (3.0 ~ 9.0)
            np.arange(10.0, 21.0, 2.0)  # 굵은 혈관 (10.0 ~ 20.0)
        ],
        threshold_val=4.0,
        min_size=400,
        second_threshold=160,
        smoothing_sigma=3.0
    )
    
    if os.path.exists(target_path):
        analyzer.analyze(target_path)
    else:
        print("파일 경로를 확인해주세요.")