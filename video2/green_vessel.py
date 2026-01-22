import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sato, frangi
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
            scale_groups = [
                range(1, 3, 1), 
                range(3, 6, 1), 
                range(6, 10, 1)
            ]
        
        self.scale_groups = scale_groups
        self.threshold_val = threshold_val
        self.min_size = min_size
        self.second_threshold = second_threshold
        self.smoothing_sigma = smoothing_sigma

    def get_first_stage_vessels(self, raw_img, roi_mask):
        green_channel = raw_img[:, :, 1]
        
        # 1. CLAHE 파라미터 상향 (흐릿한 혈관을 억지로 끌어올림)
        # clipLimit을 높이면 대비가 강해지지만 노이즈도 커지므로 2.0~3.0 사이 권장
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        inverted = cv2.bitwise_not(enhanced)
        
        # 2. Sato 필터 확률 맵 생성
        max_prob = np.zeros(raw_img.shape[:2], dtype=np.float32)
        fine_sigmas = [0.5, 1.0, 1.5] # 미세 혈관용 스케일
        all_sigmas = fine_sigmas + [s for g in self.scale_groups for s in g]
        
        for s in all_sigmas:
            vessel_prob = sato(inverted, sigmas=(s,), black_ridges=False)
            max_prob = np.maximum(max_prob, vessel_prob)

        # [중요] 3. 정규화 및 적응형 임계값 처리
        # 확률 맵을 0~255로 정규화한 뒤, 주변 영역 대비 밝은 곳을 추출
        max_prob_uint8 = cv2.normalize(max_prob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C를 사용하여 흐릿해도 주변보다 밝으면 혈관으로 인식
        vessel = cv2.adaptiveThreshold(
            max_prob_uint8, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -2 # 15는 주변 영역 크기, -2는 감도(수치를 낮출수록 더 많이 잡음)
        )
        
        # 4. 미세 노이즈 제거 (흐릿한 혈관을 유지하기 위해 최소한으로만 적용)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, kernel)
        
        binary_bool = vessel > 0
        cleaned_bool = remove_small_objects(binary_bool, min_size=100) # 미세 혈관을 위해 min_size 하향
        vessel = cleaned_bool.astype(np.uint8) * 255
        
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask), enhanced

    def get_second_stage_vessels(self, raw_img, first_stage_mask):
        lab_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB)
        a_channel = lab_img[:, :, 1]
        
        clahe_a = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
        a_enhanced = clahe_a.apply(a_channel)
        
        # 임계값 결정 및 Otsu 값 저장
        if self.second_threshold > 0:
            ret, a_thresh = cv2.threshold(a_enhanced, self.second_threshold, 255, cv2.THRESH_BINARY)
            self.computed_otsu_val = self.second_threshold # 사용자 입력값 저장
        else:
            # ret에 Otsu가 자동으로 계산한 임계값이 담깁니다.
            ret, a_thresh = cv2.threshold(a_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.computed_otsu_val = ret # 자동 계산된 Otsu 값 저장
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_seed = cv2.dilate(first_stage_mask, kernel, iterations=1)
        
        second_vessel = cv2.bitwise_or(
            cv2.bitwise_and(a_thresh, dilated_seed),
            cv2.bitwise_and(a_thresh, cv2.bitwise_not(first_stage_mask))
        )
        
        return second_vessel, a_enhanced

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
        
        # 1차 + 2차 합본 (3차 정제 제거)
        final_vessel = cv2.bitwise_or(first_stage, second_stage)
        
        # 최종 결과 시각화
        final_viz = self.create_vessel_overlay(raw, final_vessel)

        # 결과 저장
        if save_overlay:
            base_dir = os.path.dirname(img_path)
            output_dir = os.path.join(base_dir, "output")  # ← 이 줄 추가
            os.makedirs(output_dir, exist_ok=True)  # ← 이 줄 추가
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 1. 오버레이 이미지 저장
            overlay_path = os.path.join(output_dir, f"{base_name}_vessel_final_xblur.png")  # ← base_dir를 output_dir로 변경
            cv2.imwrite(overlay_path, final_viz)
            print(f"오버레이 저장 완료: {overlay_path}")
            
            # 2. 이진 마스크 이미지 저장 (흑백)
            binary_path = os.path.join(output_dir, f"{base_name}_vessel_binary_xblur.png")  # ← base_dir를 output_dir로 변경
            cv2.imwrite(binary_path, final_vessel)
            print(f"이진 마스크 저장 완료: {binary_path}")

        # 지표 계산
        if np.sum(final_vessel) > 0:
            skeleton = skeletonize(final_vessel > 0).astype(np.uint8) * 255
        else:
            skeleton = np.zeros_like(final_vessel)

        dist_map = distance_transform_edt(final_vessel > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(final_vessel > 0) / (raw.shape[0] * raw.shape[1])
        
        # 플롯 출력: 원본 - 1차 채널 - 1차 결과 - 2차 채널 - 2차 결과 - 최종 결과
        imgs = [raw, green_clahe, first_stage_overlay, a_enhanced, second_stage_overlay, final_viz]
        titles = ['Original', '1st Channel\n(Green+CLAHE)', '1st Result\n(Overlay)', 
                '2nd Channel\n(a+CLAHE)', '2nd Result\n(Overlay)', 'Final Result\n(No 3rd Stage)']
        
        plt.figure(figsize=(18, 10))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(2, 3, i+1)
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        otsu_info = f"Otsu Thr: {self.computed_otsu_val:.1f}" if self.second_threshold == 0 else f"User Thr: {self.second_threshold}"
        
        report = (f"Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px\n"
                  f"1st Thr: {self.threshold_val} | 2nd Stage ({otsu_info}) | 3rd Stage: DISABLED")
        
        plt.figtext(0.5, 0.02, report, ha="center", fontsize=12, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":10})
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    target_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff"
    
    analyzer = ThreeStageVesselSystem(
        scale_groups=[range(1, 4, 1)], # 작은 스케일에 집중
        threshold_val=2.5,             # 적응형 임계값을 쓰므로 이 값은 보조 역할
        min_size=100,                  # 작은 혈관 조각 유지
        second_threshold=150,            # 0이상의 값일 경우 그 값 사용 (default=otsu)
        smoothing_sigma=0.5            # 너무 뭉개지지 않게 낮춤
    )
        
    if os.path.exists(target_path):
        analyzer.analyze(target_path)
    else:
        print("파일 경로를 확인해주세요.")