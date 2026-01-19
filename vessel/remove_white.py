import cv2
import numpy as np
import os

# PatchMatch 라이브러리 로드
try:
    import patchmatch
    from patchmatch import inpaint as pm_inpaint
except ImportError:
    try:
        from patchmatch import patch_match as pm_inpaint
    except ImportError:
        print("에러: patchmatch 라이브러리를 찾을 수 없습니다.")
        pm_inpaint = None


def strong_specular_mask(img):
    """1번 코드의 검증된 반사광 검출 로직"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 거대 반사광 검출
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    pL = np.percentile(L, 98.5)
    large_bright = cv2.threshold(L, pL, 255, cv2.THRESH_BINARY)[1]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    low_sat = cv2.inRange(S, 0, 70)
    large_mask = cv2.bitwise_and(large_bright, low_sat)
    
    # 2. 미세 반사광 검출
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    log = cv2.Laplacian(blur, cv2.CV_32F)
    log = np.abs(log)
    log = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p_log = np.percentile(log, 95)
    small_mask = cv2.threshold(log, p_log, 255, cv2.THRESH_BINARY)[1]

    # 3. 결합 및 후처리
    combined_mask = cv2.bitwise_or(large_mask, small_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask


def separate_mask_by_size(mask, min_area_threshold=50):
    """
    마스크를 크기별로 분리
    
    Parameters:
    - mask: 전체 반사광 마스크
    - min_area_threshold: 이 값보다 작으면 미세 반사광, 크면 큰 반사광
    
    Returns:
    - large_mask: 큰 반사광 마스크
    - small_mask: 미세 반사광 마스크
    """
    # 연결된 컴포넌트 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    large_mask = np.zeros_like(mask)
    small_mask = np.zeros_like(mask)
    
    # 각 컴포넌트를 크기별로 분류 (0번은 배경이므로 제외)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        component_mask = (labels == i).astype(np.uint8) * 255
        
        if area >= min_area_threshold:
            large_mask = cv2.bitwise_or(large_mask, component_mask)
        else:
            small_mask = cv2.bitwise_or(small_mask, component_mask)
    
    return large_mask, small_mask


def force_texture_fill(img, mask):
    """1번 코드의 기존 inpaint 로직 (큰 반사광용)"""
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(mask, kernel_large, iterations=5)
    
    inpainted = cv2.inpaint(img, dilated_mask, 35, cv2.INPAINT_TELEA)
    
    blur_size = 51
    mask_blur = cv2.GaussianBlur(dilated_mask.astype(np.float32), (blur_size, blur_size), 0) / 255.0
    mask_blur = cv2.merge([mask_blur, mask_blur, mask_blur])
    
    img_f = img.astype(np.float32)
    inpainted_f = inpainted.astype(np.float32)
    
    final = img_f * (1.0 - mask_blur) + inpainted_f * mask_blur
    return np.clip(final, 0, 255).astype(np.uint8)


def patchmatch_fill(img, mask):
    """2번 코드의 PatchMatch 로직 (미세 반사광용)"""
    if pm_inpaint is None:
        print("경고: PatchMatch를 사용할 수 없어 기본 inpaint로 대체합니다.")
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    kernel_pm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(mask, kernel_pm, iterations=1)
    
    try:
        result = pm_inpaint.inpaint(img, dilated_mask, patch_size=7)
    except AttributeError:
        result = pm_inpaint(img, dilated_mask, patch_size=7)
    
    return result


def hybrid_specular_removal(image_path, save_dir, min_area_threshold=50):
    """
    반사광 크기별 혼합 처리
    
    Parameters:
    - image_path: 입력 이미지 경로
    - save_dir: 저장 디렉토리
    - min_area_threshold: 크기 구분 임계값 (픽셀 개수)
                          이 값보다 작으면 미세 반사광으로 처리
    """
    print(f"---- 처리 시작 (크기 임계값: {min_area_threshold} 픽셀) ----")
    
    # 1. 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    # 2. 전체 반사광 마스크 생성 (1번 코드 로직)
    print("반사광 검출 중...")
    full_mask = strong_specular_mask(img)
    
    # 3. 크기별로 마스크 분리
    print(f"반사광을 크기별로 분류 중 (임계값: {min_area_threshold})...")
    large_mask, small_mask = separate_mask_by_size(full_mask, min_area_threshold)
    
    # 통계 출력
    large_count = cv2.countNonZero(large_mask)
    small_count = cv2.countNonZero(small_mask)
    print(f"  - 큰 반사광 픽셀: {large_count}")
    print(f"  - 미세 반사광 픽셀: {small_count}")
    
    # 4. 큰 반사광 처리 (1번 코드 방식)
    result = img.copy()
    if large_count > 0:
        print("큰 반사광 처리 중 (기존 inpaint)...")
        result = force_texture_fill(result, large_mask)
    
    # 5. 미세 반사광 처리 (2번 코드 방식)
    if small_count > 0:
        print("미세 반사광 처리 중 (PatchMatch)...")
        result = patchmatch_fill(result, small_mask)
    
    # 6. 저장
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    out_img_path = os.path.join(save_dir, f"{base}_hybrid_{min_area_threshold}.tiff")
    out_full_mask_path = os.path.join(save_dir, f"{base}_mask_full.tiff")
    out_large_mask_path = os.path.join(save_dir, f"{base}_mask_large.tiff")
    out_small_mask_path = os.path.join(save_dir, f"{base}_mask_small.tiff")
    
    cv2.imwrite(out_img_path, result, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    cv2.imwrite(out_full_mask_path, full_mask)
    cv2.imwrite(out_large_mask_path, large_mask)
    cv2.imwrite(out_small_mask_path, small_mask)
    
    print("---- 처리 완료 ----")
    print(f"결과 이미지: {out_img_path}")
    print(f"전체 마스크: {out_full_mask_path}")
    print(f"큰 반사광 마스크: {out_large_mask_path}")
    print(f"미세 반사광 마스크: {out_small_mask_path}")
    
    return out_img_path


# ===============================
# 실행 및 테스트
# ===============================
if __name__ == "__main__":
    image_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned.tiff"
    save_dir = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI"
    
    # 단일 threshold로 실행
    threshold = 97
    
    print(f"{'='*60}")
    print(f"임계값 {threshold}으로 처리 중...")
    print(f"{'='*60}")
    hybrid_specular_removal(image_path, save_dir, min_area_threshold=threshold)