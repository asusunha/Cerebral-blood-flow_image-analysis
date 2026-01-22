import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import io, img_as_float, exposure
from skimage.filters import meijering, threshold_triangle
from skimage.morphology import remove_small_objects, skeletonize, disk

def analyze_vessel_ultra_thin(file_path):
    if not os.path.exists(file_path):
        return f"파일을 찾을 수 없습니다: {file_path}"
    
    # 1. 이미지 로드 및 기초 전처리
    raw_img = io.imread(file_path)
    if len(raw_img.shape) == 3:
        raw_img = raw_img[:, :, 1]
    
    img = img_as_float(raw_img)
    if np.mean(img) > 0.5:
        img = 1.0 - img
    
    img_8bit = (img * 255).astype(np.uint8)

    # 2. 전처리: 대비 증폭은 유지하되 배경 조직 억제 (CLAHE 하향)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    img_clahe = clahe.apply(img_8bit)
    img_blurred = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    img_final_pre = img_as_float(img_blurred)

    # 3. 선형 구조물 강조 (초미세 Sigma 설정)
    # 얇은 혈관(1~5px)에 극도로 민감하게 반응하도록 설정
    vessels = meijering(img_final_pre, sigmas=np.arange(1, 6, 1), black_ridges=False)
    vessels_norm = exposure.rescale_intensity(vessels)

    # 4. 극한의 이진화: 혈관 중심선(Peak)만 남기기
    try:
        thresh = threshold_triangle(vessels_norm)
        # 임계값 가중치를 4.5~5.5 수준으로 높여 혈관을 극도로 얇게 깎음
        binary = vessels_norm > (thresh * 5.0) 
    except:
        # 실패 시 최상위 2% 신호만 채택
        binary = vessels_norm > np.percentile(vessels_norm, 98)

    # 5. 후처리: 두께를 부풀리는 Closing 연산 완전 제거
    # binary_closing을 수행하지 않음으로써 선의 날카로움 유지
    
    # 선이 얇아져서 끊길 수 있으므로 min_size를 현실적으로 하향 조정
    cleaned = remove_small_objects(binary, min_size=150)
    skeleton = skeletonize(cleaned)

    return {
        "original": img_final_pre,
        "vessel_map": vessels_norm,
        "binary": cleaned,
        "skeleton": skeleton,
        "raw_max": np.max(vessels),
    }

# --- 실행 및 시각화 ---
file_path = r'D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff'
results = analyze_vessel_ultra_thin(file_path)

if isinstance(results, dict):
    plt.figure(figsize=(24, 6))
    imgs = [results['original'], results['vessel_map'], results['binary'], results['skeleton']]
    titles = ['Preprocessed', 'Ultra-thin Vessel Map', 'Binary (Super Tight)', 'Skeleton']
    
    for i in range(4):
        plt.subplot(1, 4, i+1)
        if i == 1:
            plt.imshow(np.log1p(imgs[i] * 100), cmap='magma')
        else:
            plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()