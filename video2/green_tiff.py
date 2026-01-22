import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
import os

class FinalVesselSystem:
    # 초기화 파라미터에 min_size 추가 및 threshold 기본값 상향
    def __init__(self, sato_sigmas=range(1, 5, 1), threshold_val=0.0, min_size=300):
        self.sato_sigmas = sato_sigmas
        self.threshold_val = threshold_val # 노이즈가 많으면 이 값을 올리세요 (0.05 ~ 0.15)
        self.min_size = min_size         # 자잘한 점이 많으면 이 값을 올리세요 (200 ~ 500)

    def get_clean_vessels(self, raw_img, roi_mask):
        """[Sato 필터 기반] 노이즈 제거가 강화된 혈관 추출 로직"""
        # 1. A-Channel 추출
        lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        
        # 2. 대비 강화 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # clipLimit을 3.0 -> 2.0으로 낮춰 노이즈 증폭 감소
        enhanced = clahe.apply(a_channel)
        
        # [추가됨] 2.5. 전처리 블러링: Sato 필터 적용 전 노이즈 스무딩
        # 커널 크기 (5,5)는 상황에 따라 조절 가능
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 3. Sato 필터 적용 (혈관 확률맵 생성)
        # black_ridges=False는 밝은 배경의 어두운 혈관이 아니라, 어두운 배경의 밝은 혈관을 찾을 때 사용
        # A-channel은 혈관이 밝게 나오므로 False가 맞습니다.
        vessel_probability = sato(blurred, sigmas=self.sato_sigmas, black_ridges=False)
        
        # 4. 이진화: 확률맵을 0과 255로 변환
        # threshold_val보다 높은 확률을 가진 픽셀만 흰색으로 만듦
        binary_vessel = (vessel_probability > self.threshold_val).astype(np.uint8) * 255
        
        # 5. 강력한 후처리: 작은 노이즈 덩어리 제거
        # min_size 픽셀보다 작은 덩어리는 모두 지워버림
        binary_bool = binary_vessel > 0
        cleaned_bool = remove_small_objects(binary_bool, min_size=self.min_size)
        vessel = cleaned_bool.astype(np.uint8) * 255
        
        # ROI 마스크 적용 (혹시 모를 외곽 노이즈 제거)
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask)

    def analyze(self, img_path):
        raw = cv2.imread(img_path)
        if raw is None: return
        
        # ROI가 필요하다면 이전 코드의 get_color_roi를 복사해서 쓰세요.
        # 현재는 전체 이미지를 대상으로 합니다.
        roi_mask = np.ones(raw.shape[:2], dtype=np.uint8) * 255
        
        # 혈관 추출
        vessel = self.get_clean_vessels(raw, roi_mask)
        
        # 스켈레톤(중심선) 추출 - 혈관이 조금이라도 있어야 수행
        if np.sum(vessel) > 0:
            skeleton = skeletonize(vessel > 0).astype(np.uint8) * 255
        else:
            skeleton = np.zeros_like(vessel)

        # 지표 계산
        dist_map = distance_transform_edt(vessel > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(vessel > 0) / (raw.shape[0] * raw.shape[1])
        
        # 시각화
        lab = cv2.cvtColor(raw, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        
        final_viz = raw.copy()
        # 원본에 녹색 오버레이 (투명도 적용)
        mask_indices = vessel > 0
        if np.any(mask_indices):
             final_viz[mask_indices] = cv2.addWeighted(raw[mask_indices], 0.3, 
                                                       np.full_like(raw[mask_indices], (0, 255, 0)), 0.7, 0)

        imgs = [raw, a_channel, vessel, skeleton, final_viz]
        titles = ['Original', 'A-Channel', 'Cleaned Vessel Map', 'Skeleton', 'Overlay']
        
        plt.figure(figsize=(20, 8))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(1, 5, i+1)
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255) # 흑백 이미지 대비 명확히
            plt.title(title)
            plt.axis('off')
        
        report = f"Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px\nParams: th={self.threshold_val}, min_size={self.min_size}"
        plt.figtext(0.5, 0.05, report, ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.8, "pad":10})
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # ▼▼▼ 여기가 핵심 조절 파트입니다 ▼▼▼
    # threshold_val을 올릴수록 배경이 깨끗해집니다.
    # min_size를 올릴수록 자잘한 점들이 사라집니다.
    analyzer = FinalVesselSystem(sato_sigmas=range(1, 15, 1), threshold_val=2.5, min_size=200)
    
    target_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Baseline_ROI\base_f00_original_2.tiff"
    if os.path.exists(target_path):
        analyzer.analyze(target_path)
    else:
        print("파일 경로 확인 필요")