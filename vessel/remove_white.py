import cv2
import numpy as np
import os


def strong_specular_mask(img):
    # LAB 밝기 기반
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    pL = np.percentile(L, 98.5)
    bright = (L > pL).astype(np.uint8) * 255

    # HSV 채도 기반 (반사광은 채도 낮음)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    low_sat = cv2.inRange(S, 0, 60)

    # Laplacian 기반 날카로운 하이라이트 검출
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    log = cv2.Laplacian(blur, cv2.CV_32F)
    log = np.abs(log)
    log = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p_log = np.percentile(log, 85)
    sharp = (log > p_log).astype(np.uint8) * 255

    # 마스크 결합
    mask = cv2.bitwise_and(bright, low_sat)
    mask = cv2.bitwise_or(mask, sharp)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def force_texture_fill(img, mask):
    # 1. 마스크 확장
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(mask, kernel_large, iterations=5)

    # 2. 강제 인페인팅
    inpainted = cv2.inpaint(img, dilated_mask, 35, cv2.INPAINT_TELEA)

    # 3. 알파 블렌딩
    blur_size = 51
    mask_blur = cv2.GaussianBlur(dilated_mask.astype(np.float32), (blur_size, blur_size), 0) / 255.0
    mask_blur = cv2.merge([mask_blur, mask_blur, mask_blur])

    img_f = img.astype(np.float32)
    inpainted_f = inpainted.astype(np.float32)

    final = img_f * (1.0 - mask_blur) + inpainted_f * mask_blur
    return np.clip(final, 0, 255).astype(np.uint8)


def remove_specular(image_path, save_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")

    # 1. 반사광 마스크 생성
    mask = strong_specular_mask(img)

    # 2. 제거 수행
    result = force_texture_fill(img, mask)

    # 3. 저장
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.join(save_dir, base + "_final.tiff")
    out_mask_path = os.path.join(save_dir, base + "_mask.tiff")

    cv2.imwrite(out_img_path, result)
    cv2.imwrite(out_mask_path, mask)

    print("---- 처리 완료 ----")
    print("결과:", out_img_path)
    print("마스크:", out_mask_path)

    return out_img_path, out_mask_path


# ===============================
# 실행
# ===============================
image_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned.tiff"
save_dir = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI"

remove_specular(image_path, save_dir)
