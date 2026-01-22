import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로
img_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97_vessels_removed.png"

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

# ============ 3. 모든 채널 출력 ============
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
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.ravel()

for idx, (name, channel) in enumerate(channels.items()):
    axes[idx].imshow(channel, cmap='gray')
    axes[idx].set_title(f'{name}\n범위: {channel.min()}-{channel.max()}', fontsize=9)
    axes[idx].axis('off')

# 빈 공간 제거
for idx in range(len(channels), len(axes)):
    axes[idx].axis('off')

plt.suptitle('모든 채널 미리보기', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 사용 가능한 채널 목록 출력
print("\n사용 가능한 채널:")
for key in channels.keys():
    print(f"  {key}")