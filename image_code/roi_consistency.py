import cv2
import numpy as np
import os
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import directed_hausdorff

def get_clean_vessel_network(image_path, min_vessel_size=200):
    """
    이미지에서 노이즈를 제거하고 주요 혈관의 1픽셀 골격만 추출합니다.
    """
    # 1. 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")

    # 2. 노이즈 억제 (Gaussian Blur)
    # 배경의 미세한 점들을 뭉개서 Frangi 필터가 반응하지 않게 합니다.
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Frangi Filter (탐색 범위를 2~6으로 높여 굵은 혈관 위주로 추출)
    # sigmas 범위가 커질수록 미세 노이즈에 덜 민감해집니다.
    vessels = frangi(blurred, sigmas=range(2, 7, 1))
    
    # 4. 엄격한 이진화 (Thresholding)
    # 평균보다 훨씬 밝은(확실한) 혈관 영역만 선택합니다.
    binary = vessels > (np.mean(vessels) + np.std(vessels) * 2.5)
    
    # 5. 소형 객체 제거 (핵심 단계)
    # 픽셀 연결 면적이 min_vessel_size보다 작은 파편들은 모두 지웁니다.
    cleaned = remove_small_objects(binary, min_size=min_vessel_size)
    
    # 6. 골격화 (Skeletonization)
    # 굵기를 무시하고 중심 네트워크만 추출합니다.
    skeleton = skeletonize(cleaned).astype(np.uint8) * 255
    return skeleton

def calculate_average_hausdorff(coords1, coords2):
    """
    최대 거리가 아닌 평균적인 네트워크 거리를 계산합니다.
    """
    from scipy.spatial.distance import cdist
    if len(coords1) == 0 or len(coords2) == 0: return float('inf')
    
    # 각 점에서 가장 가까운 점까지의 거리들의 평균
    dists_1to2 = np.min(cdist(coords1, coords2), axis=1)
    dists_2to1 = np.min(cdist(coords2, coords1), axis=1)
    return (np.mean(dists_1to2) + np.mean(dists_2to1)) / 2

# --- 설정 및 실행 ---

base_path = r"D:\VIDEO\M-24_captured\Video_00056_00060\Aligned_ROI_v5_drag"
baseline_file = "Baseline_Video_00056_00001_0s_1356_308_1338_1195.tiff"
aligned_file = "Aligned_Video_00056_00060_0s_1356_308_1338_1195.tiff"

print("Step 1: Baseline 혈관 추출 중...")
skel_base = get_clean_vessel_network(os.path.join(base_path, baseline_file))

print("Step 2: Aligned 혈관 추출 중...")
skel_aligned = get_clean_vessel_network(os.path.join(base_path, aligned_file))

# 좌표 추출
coords_base = np.column_stack(np.where(skel_base > 0))
coords_aligned = np.column_stack(np.where(skel_aligned > 0))

# 지표 계산
max_dist = max(directed_hausdorff(coords_base, coords_aligned)[0], 
               directed_hausdorff(coords_aligned, coords_base)[0])
avg_dist = calculate_average_hausdorff(coords_base, coords_aligned)

print(f"\n--- 분석 결과 ---")
print(f"최대 구조 거 (Max Hausdorff): {max_dist:.2f} px")
print(f"평균 구조 거리 (Avg Hausdorff): {avg_dist:.2f} px")

# 시각화 결과 저장
h, w = skel_base.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)
vis[skel_base > 0] = [255, 0, 0]    # Baseline: 파랑
vis[skel_aligned > 0] = [0, 0, 255] # Aligned: 빨강
# 두 선이 겹치는 곳은 보라색/흰색으로 보임

save_path = os.path.join(base_path, "Cleaned_Comparison_v2.png")
cv2.imwrite(save_path, vis)
print(f"\n결과 이미지가 저장되었습니다: {save_path}")