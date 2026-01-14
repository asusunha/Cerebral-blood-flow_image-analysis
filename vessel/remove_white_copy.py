import cv2
import numpy as np
import os


# =========================================
# 1. 반사광 + Halo까지 포함하는 강력한 마스크
# =========================================
def strong_specular_mask(img):
    # LAB 밝기
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    pL = np.percentile(L, 96.5)     # ★ 기존 98.5 → halo 포함
    bright = (L > pL).astype(np.uint8) * 255

    # HSV 채도 (반사광/halo는 채도 낮음)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    low_sat = cv2.inRange(S, 0, 70)   # ★ 60 → 70 (halo 회색도 포함)

    # Laplacian sharpness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    log = cv2.Laplacian(blur, cv2.CV_32F)
    log = np.abs(log)
    log = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p_log = np.percentile(log, 70)    # ★ 85 → 70 (halo edge 포함)
    sharp = (log > p_log).astype(np.uint8) * 255

    # 마스크 결합
    mask = cv2.bitwise_and(bright, low_sat)
    mask = cv2.bitwise_or(mask, sharp)

    # Halo 연결시키기
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))   # ★ 3→7
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# =========================================
# 2. Halo 바깥 조직으로 강제 인페인팅
# =========================================
def force_texture_fill(img, mask):
    # Halo까지 완전히 제거되도록 확장
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_mask = cv2.dilate(mask, kernel_large, iterations=6)   # ★ 더 멀리 제거

    # 멀리서 참조하는 인페인팅
    inpainted = cv2.inpaint(img, dilated_mask, 60, cv2.INPAINT_TELEA)  # ★ 35 → 60

    # 부드러운 블렌딩 (halo 섞이지 않게 넓게)
    blur_size = 81
    mask_blur = cv2.GaussianBlur(dilated_mask.astype(np.float32),
                                (blur_size, blur_size), 0) / 255.0
    mask_blur = cv2.merge([mask_blur, mask_blur, mask_blur])

    img_f = img.astype(np.float32)
    inpainted_f = inpainted.astype(np.float32)

    final = img_f * (1.0 - mask_blur) + inpainted_f * mask_blur
    return np.clip(final, 0, 255).astype(np.uint8), dilated_mask


# =========================================
# 3. 전체 실행 함수
# =========================================
def remove_specular(image_path, save_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")

    # 1. 반사광 + halo 마스크
    mask = strong_specular_mask(img)

    # 2. halo 바깥 조직으로 복원
    result, dilated_mask = force_texture_fill(img, mask)

    # 3. 저장
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.join(save_dir, base + "_final.tiff")
    out_mask_path = os.path.join(save_dir, base + "_mask.tiff")
    out_dilated_path = os.path.join(save_dir, base + "_mask_dilated.tiff")

    cv2.imwrite(out_img_path, result)
    cv2.imwrite(out_mask_path, mask)
    cv2.imwrite(out_dilated_path, dilated_mask)

    print("---- 처리 완료 ----")
    print("결과:", out_img_path)
    print("마스크(core+halo):", out_mask_path)
    print("확장마스크(인페인팅영역):", out_dilated_path)

    return out_img_path, out_mask_path, out_dilated_path


# =========================================
# 실행
# =========================================
image_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned.tiff"
save_dir = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI"

remove_specular(image_path, save_dir)
