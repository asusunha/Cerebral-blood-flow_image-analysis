import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
from skimage import exposure

def process_combined_skeleton():
    # 1. 기본 경로 설정
    base_path = r"D:\VIDEO\M-24\result_2115.89um_2132.81um\Baseline_1057.94um_1066.41um"
    
    # 2. 각 사분면 이미지 로드
    # q1: 좌상, q2: 우상, q3: 좌하, q4: 우하
    q1 = cv2.imread(os.path.join(base_path, "Quadrant_1", "base_f00_q1.tiff"))
    q2 = cv2.imread(os.path.join(base_path, "Quadrant_2", "base_f00_q2.tiff"))
    q3 = cv2.imread(os.path.join(base_path, "Quadrant_3", "base_f00_q3.tiff"))
    q4 = cv2.imread(os.path.join(base_path, "Quadrant_4", "base_f00_q4.tiff"))

    if q1 is None or q2 is None or q3 is None or q4 is None:
        print("❌ 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        return

    # 3. 이미지 합치기 (Concatenation)
    top_row = cv2.hconcat([q1, q2])      # 좌상 + 우상
    bottom_row = cv2.hconcat([q3, q4])   # 좌하 + 우하
    combined_roi = cv2.vconcat([top_row, bottom_row]) # 상단 + 하단

    # 4. 혈관 강조 및 골격화 전처리
    gray = cv2.cvtColor(combined_roi, cv2.COLOR_BGR2GRAY)
    
    # Sato 필터: 미세 혈관 추출에 최적화
    vesselness = sato(gray, sigmas=range(1, 10, 2), black_ridges=False)
    vesselness_rescaled = exposure.rescale_intensity(vesselness, out_range=(0, 255)).astype(np.uint8)
    
    # 이진화 (Adaptive Threshold로 미세혈관 보존)
    binary = cv2.adaptiveThreshold(vesselness_rescaled, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 17, -2)
    
    # 노이즈 제거 및 골격화 (1픽셀 두께)
    cleaned = remove_small_objects(binary.astype(bool), min_size=50)
    skeleton = skeletonize(cleaned)

    # 5. 결과 출력
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title("Combined Original ROI", fontsize=15)
    plt.imshow(cv2.cvtColor(combined_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("1-pixel Vessel Skeleton", fontsize=15)
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 픽셀 기반 총 길이 계산
    total_length = np.sum(skeleton)
    print(f"✅ 분석 완료! 통합된 혈관의 총 픽셀 길이: {total_length} px")

# 실행
process_combined_skeleton()