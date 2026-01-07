import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def detect_glare_and_gray(img_bgr, threshold=215, kernel_size=7):
    """
    미세 점광, 고휘도 반사광, 그리고 채도가 낮은 밝은 회색 픽셀을 감지합니다.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. 탑햇 변환: 주변보다 밝은 미세 점/패턴 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask_tophat = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY) # 감도 상향 (30->20)

    # 2. 전역 임계값: 아주 밝은 픽셀 제거
    _, mask_global = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 3. [추가] 밝은 회색 필터링 (HSV 공간 활용)
    # S(채도)가 낮고(회색조), V(명도)가 높은(밝은) 영역을 타겟팅
    # 보통 천(Drape)이나 밝은 금속성 회색이 여기에 해당합니다.
    lower_gray = np.array([0, 0, 150])    # 명도 150 이상의 밝은 영역
    upper_gray = np.array([180, 55, 255]) # 채도 55 이하의 무채색 영역
    mask_gray_zone = cv2.inRange(hsv, lower_gray, upper_gray)

    # 모든 마스크 통합
    combined = cv2.bitwise_or(mask_tophat, mask_global)
    combined = cv2.bitwise_or(combined, mask_gray_zone)
    
    # 4. 마스크 확장 (주변부Halo 및 격자무늬 메우기)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.dilate(combined, dilate_kernel, iterations=1)
    
    return refined_mask

def verify_alignment_advanced_mask(baseline_path, aligned_path):
    img_base = cv2.imread(baseline_path)
    img_aligned = cv2.imread(aligned_path)

    if img_base is None or img_aligned is None: return

    # 왼쪽 350픽셀 크롭
    img_base = img_base[:, 350:]
    img_aligned = img_aligned[:, 350:]

    # 고급 마스크 추출 (밝은 회색 포함)
    mask_base = detect_glare_and_gray(img_base)
    mask_aligned = detect_glare_and_gray(img_aligned)

    # 공통 제외 영역
    combined_mask = cv2.bitwise_or(mask_base, mask_aligned)
    valid_mask = cv2.bitwise_not(combined_mask)

    # SSIM 계산용 그레이스케일
    gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)

    score_full, diff = ssim(gray_base, gray_aligned, full=True)

    # 유효 영역 평균 SSIM
    valid_ssim_values = diff[valid_mask > 0]
    final_score = valid_ssim_values.mean() if valid_ssim_values.size > 0 else 0

    # 이진 마스크 저장
    save_dir = os.path.dirname(aligned_path)
    glare_mask_path = os.path.join(save_dir, "glare_mask.png")
    cv2.imwrite(glare_mask_path, combined_mask)

    # 시각화
    vis_excluded = img_aligned.copy()
    vis_excluded[combined_mask > 0] = [255, 0, 255] # 제외 영역 핑크색

    print(f"\n[밝은 회색 및 노이즈 제거 후 결과]")
    print("-" * 45)
    print(f"📊 최종 유효 영역 SSIM: {final_score:.4f}")
    print(f"📍 제외된 노이즈 면적: {np.sum(combined_mask > 0)/(gray_base.size)*100:.2f}%")
    print("-" * 45)

    cv2.imshow("Excluded Areas (Pink)", vis_excluded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ROOT_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030\Aligned_ROI_v4'
    base_p = os.path.join(ROOT_DIR, 'Baseline_Video_00056_00001_0s.tiff')
    align_p = os.path.join(ROOT_DIR, 'Aligned_Video_00056_00030_0s.tiff')
    verify_alignment_advanced_mask(base_p, align_p)