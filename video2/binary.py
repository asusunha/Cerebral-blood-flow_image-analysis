import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.morphology import opening, closing, disk, remove_small_objects
from scipy import ndimage

# 이미지 경로
img_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff"

# 이미지 로드
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

print("=" * 60)
print("혈관 이미지 다중 채널 분석 도구")
print("=" * 60)

# BGR to RGB 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ============ 1. 다양한 색공간 채널 추출 ============
# RGB 채널
r_channel = img_rgb[:, :, 0]
g_channel = img_rgb[:, :, 1]
b_channel = img_rgb[:, :, 2]

# 그레이스케일
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Lab 색공간
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
l_channel = img_lab[:, :, 0]
a_channel = img_lab[:, :, 1]
b_channel_lab = img_lab[:, :, 2]

# HSV 색공간
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_channel = img_hsv[:, :, 0]
s_channel = img_hsv[:, :, 1]
v_channel = img_hsv[:, :, 2]

# ============ 2. CLAHE 적용 (대비 향상) ============
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)
g_clahe = clahe.apply(g_channel)
l_clahe = clahe.apply(l_channel)

# ============ 3. 모든 채널 출력 및 선택 ============
channels = {
    '1. RGB - Red': r_channel,
    '2. RGB - Green': g_channel,
    '3. RGB - Blue': b_channel,
    '4. Grayscale': gray,
    '5. Lab - L (Lightness)': l_channel,
    '6. Lab - A (Green-Red)': a_channel,
    '7. Lab - B (Blue-Yellow)': b_channel_lab,
    '8. HSV - H (Hue)': h_channel,
    '9. HSV - S (Saturation)': s_channel,
    '10. HSV - V (Value)': v_channel,
    '11. Gray + CLAHE': gray_clahe,
    '12. Green + CLAHE': g_clahe,
    '13. L + CLAHE': l_clahe
}

# 모든 채널 미리보기
fig1, axes1 = plt.subplots(3, 5, figsize=(20, 12))
axes1 = axes1.ravel()

for idx, (name, channel) in enumerate(channels.items()):
    axes1[idx].imshow(channel, cmap='gray')
    axes1[idx].set_title(f'{name}\n범위: {channel.min()}-{channel.max()}', fontsize=9)
    axes1[idx].axis('off')

# 빈 공간 제거
for idx in range(len(channels), len(axes1)):
    axes1[idx].axis('off')

plt.suptitle('모든 채널 미리보기 - 분석할 채널을 선택하세요', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 채널 선택
print("\n사용 가능한 채널:")
for key in channels.keys():
    print(f"  {key}")

channel_num = int(input("\n분석할 채널 번호를 입력하세요 (1-13): "))
channel_names = list(channels.keys())
selected_name = channel_names[channel_num - 1]
selected_channel = channels[selected_name]

print(f"\n선택된 채널: {selected_name}")
print(f"채널 범위: {selected_channel.min()} - {selected_channel.max()}")

# ============ 4. 임계값 입력 ============
threshold = float(input(f"임계값을 입력하세요 (추천: {selected_channel.mean():.1f}): "))

# ============ 5. 다양한 이진화 기법 적용 ============
# 기본 이진화
_, binary_basic = cv2.threshold(selected_channel, threshold, 255, cv2.THRESH_BINARY)

# 반전 이진화 (혈관이 어두운 경우)
_, binary_inv = cv2.threshold(selected_channel, threshold, 255, cv2.THRESH_BINARY_INV)

# Otsu 자동 임계값
_, binary_otsu = cv2.threshold(selected_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive threshold (지역적 임계값)
binary_adaptive = cv2.adaptiveThreshold(selected_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

# ============ 6. Morphological Operations ============
kernel = disk(2)

# Opening (노이즈 제거)
binary_opening = opening(binary_basic > 0, kernel).astype(np.uint8) * 255

# Closing (구멍 메우기)
binary_closing = closing(binary_basic > 0, kernel).astype(np.uint8) * 255

# Opening + Closing
binary_morph = closing(opening(binary_basic > 0, kernel), kernel).astype(np.uint8) * 255

# ============ 7. 고급 필터링 ============
# Frangi filter (혈관 강조 - 관 형태 검출)
try:
    from skimage.filters import frangi
    frangi_enhanced = frangi(selected_channel, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15, black_ridges=False)
    frangi_enhanced = (frangi_enhanced * 255).astype(np.uint8)
    _, binary_frangi = cv2.threshold(frangi_enhanced, threshold/5, 255, cv2.THRESH_BINARY)
    use_frangi = True
except:
    use_frangi = False
    print("Warning: Frangi filter를 사용할 수 없습니다.")

# Hessian-based enhancement
from skimage.filters import meijering, sato
meijering_enhanced = meijering(selected_channel, sigmas=range(1, 10, 2), black_ridges=False)
meijering_enhanced = (meijering_enhanced * 255).astype(np.uint8)
_, binary_meijering = cv2.threshold(meijering_enhanced, threshold/5, 255, cv2.THRESH_BINARY)

# ============ 8. 결과 출력 ============
num_results = 11 if use_frangi else 10
fig2, axes2 = plt.subplots(3, 4, figsize=(20, 15))
axes2 = axes2.ravel()

results = [
    ('원본 이미지', img_rgb, 'color'),
    (f'선택 채널: {selected_name}', selected_channel, 'gray'),
    (f'기본 이진화 (임계값: {threshold})', binary_basic, 'gray'),
    ('반전 이진화', binary_inv, 'gray'),
    (f'Otsu 자동 임계값', binary_otsu, 'gray'),
    ('Adaptive Threshold', binary_adaptive, 'gray'),
    ('Morphology: Opening', binary_opening, 'gray'),
    ('Morphology: Closing', binary_closing, 'gray'),
    ('Morphology: Open+Close', binary_morph, 'gray'),
    ('Meijering Filter', binary_meijering, 'gray'),
]

if use_frangi:
    results.append(('Frangi Filter (혈관 강조)', binary_frangi, 'gray'))

for idx, (title, img_data, cmap_type) in enumerate(results):
    if cmap_type == 'color':
        axes2[idx].imshow(img_data)
    else:
        axes2[idx].imshow(img_data, cmap='gray')
    
    # 흰색 픽셀 비율 계산 (이진 이미지만)
    if idx >= 2 and isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8:
        white_ratio = (img_data == 255).sum() / img_data.size * 100
        axes2[idx].set_title(f'{title}\n흰색: {white_ratio:.2f}%', fontsize=10)
    else:
        axes2[idx].set_title(title, fontsize=10)
    
    axes2[idx].axis('off')

# 빈 공간 제거
for idx in range(len(results), len(axes2)):
    axes2[idx].axis('off')

plt.suptitle(f'다양한 혈관 분할 기법 비교 - 임계값: {threshold}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============ 9. 통계 출력 ============
print("\n" + "=" * 60)
print("처리 결과 통계")
print("=" * 60)
print(f"이미지 크기: {img.shape}")
print(f"선택 채널: {selected_name}")
print(f"채널 범위: {selected_channel.min()} - {selected_channel.max()}")
print(f"사용된 임계값: {threshold}")
print(f"채널 평균값: {selected_channel.mean():.2f}")
print(f"채널 표준편차: {selected_channel.std():.2f}")
print("\n각 기법별 혈관 영역 비율:")
print(f"  기본 이진화: {(binary_basic == 255).sum() / binary_basic.size * 100:.2f}%")
print(f"  반전 이진화: {(binary_inv == 255).sum() / binary_inv.size * 100:.2f}%")
print(f"  Otsu: {(binary_otsu == 255).sum() / binary_otsu.size * 100:.2f}%")
print(f"  Adaptive: {(binary_adaptive == 255).sum() / binary_adaptive.size * 100:.2f}%")
print(f"  Morphology: {(binary_morph == 255).sum() / binary_morph.size * 100:.2f}%")
if use_frangi:
    print(f"  Frangi Filter: {(binary_frangi == 255).sum() / binary_frangi.size * 100:.2f}%")
print(f"  Meijering Filter: {(binary_meijering == 255).sum() / binary_meijering.size * 100:.2f}%")
print("=" * 60)