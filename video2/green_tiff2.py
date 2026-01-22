import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
import os

class MultiScaleVesselSystem:
    def __init__(self, scale_groups=None, threshold_val=3.0, min_size=300):
        """
        scale_groups: 각 스케일 그룹 리스트 (예: [range(1,3), range(3,6), range(6,10)])
        threshold_val: MAX 확률 맵에 적용할 단일 임계값
        min_size: 제거할 작은 객체의 최소 크기 (노이즈 제거용)
        """
        if scale_groups is None:
            scale_groups = [
                range(1, 3, 1),   # 매우 얇은 혈관
                range(3, 6, 1),   # 중간 혈관
                range(6, 10, 1)   # 굵은 혈관
            ]
        
        self.scale_groups = scale_groups
        self.threshold_val = threshold_val
        self.min_size = min_size

    def get_max_vessels(self, raw_img, roi_mask):
        """이미지 반전 후 Multi-scale Sato 필터의 최댓값(MAX) 기반 추출"""
        # 1. Green 채널 추출
        green_channel = raw_img[:, :, 1]
        
        # 2. 전처리: CLAHE 후 이미지 반전 (어두운 혈관을 밝게 변환)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        inverted = cv2.bitwise_not(enhanced)
        
        # 3. 노이즈 억제를 위한 가우시안 블러
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        
        # 4. Multi-scale Sato 적용 및 Max Probability 계산
        max_prob = np.zeros(raw_img.shape[:2], dtype=np.float32)
        for sigmas in self.scale_groups:
            # 반전된 이미지(밝은 혈관)이므로 black_ridges=False 사용
            vessel_prob = sato(blurred, sigmas=sigmas, black_ridges=False)
            max_prob = np.maximum(max_prob, vessel_prob)
        
        # 5. 이진화
        vessel = (max_prob > self.threshold_val).astype(np.uint8) * 255
        
        # 6. 후처리 (Morphology)
        # Closing: 혈관 내부의 작은 구멍을 메우고 끊어진 곳 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, kernel)
        
        # 작은 노이즈 제거
        binary_bool = vessel > 0
        cleaned_bool = remove_small_objects(binary_bool, min_size=self.min_size)
        vessel = cleaned_bool.astype(np.uint8) * 255
        
        # ROI 마스크 적용
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask)

    def analyze(self, img_path, save_overlay=True):
        raw = cv2.imread(img_path)
        if raw is None: 
            print(f"이미지를 불러올 수 없습니다: {img_path}")
            return
        
        roi_mask = np.ones(raw.shape[:2], dtype=np.uint8) * 255
        
        # 혈관 추출 (MAX 방식)
        vessel = self.get_max_vessels(raw, roi_mask)
        
        # 스켈레톤 추출
        if np.sum(vessel) > 0:
            skeleton = skeletonize(vessel > 0).astype(np.uint8) * 255
        else:
            skeleton = np.zeros_like(vessel)

        # 지표 계산
        dist_map = distance_transform_edt(vessel > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(vessel > 0) / (raw.shape[0] * raw.shape[1])
        
        # 시각화 (Overlay 제작)
        green_channel = raw[:, :, 1]
        final_viz = raw.copy()
        mask_indices = vessel > 0
        if np.any(mask_indices):
            # 혈관 영역을 연두색(0, 255, 0)으로 반투명하게 합성
            final_viz[mask_indices] = cv2.addWeighted(
                raw[mask_indices], 0.4, 
                np.full_like(raw[mask_indices], (0, 255, 0)), 0.6, 0
            )

        # 결과 저장
        if save_overlay:
            base_dir = os.path.dirname(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_vessel_max.png")
            cv2.imwrite(output_path, final_viz)
            print(f"결과 저장 완료: {output_path}")

        # 플롯 출력
        imgs = [raw, green_channel, vessel, skeleton, final_viz]
        titles = ['Original', 'Green Channel', 'Vessel (MAX)', 'Skeleton', 'Overlay']
        
        plt.figure(figsize=(22, 7))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(1, 5, i+1)
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        report = (f"Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px\n"
                  f"Threshold: {self.threshold_val} | Min Size: {self.min_size}")
        plt.figtext(0.5, 0.02, report, ha="center", fontsize=12, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":10})
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 파일 경로 설정
    target_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff"
    
    # 시스템 설정 (MAX 방식 전용)
    analyzer = MultiScaleVesselSystem(
        scale_groups=[
            range(1, 3, 1), 
            range(3, 6, 1), 
            range(6, 12, 1)  # 굵은 혈관 두께를 고려하여 범위를 약간 확장
        ],
        threshold_val=4.0,   # 배경 노이즈가 많다면 3.5 ~ 5.0 사이로 조정하세요
        min_size=400         # 너무 작은 부스러기 노이즈를 제거하기 위해 상향
    )
    
    if os.path.exists(target_path):
        analyzer.analyze(target_path)
    else:
        print("파일 경로를 확인해주세요.")