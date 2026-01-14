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
    # 1. 마스크 최적화 (경계의 미세한 반사광을 잡기 위해 팽창)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel_small, iterations=2)

    # 2. 텍스처 복제 소스 생성
    # 마스크 영역을 채우기 위해 전체 이미지를 약간 이동(Shift)시켜서 
    # 마스크 바로 옆의 '진짜 텍스처'를 소스로 활용합니다.
    rows, cols = img.shape[:2]
    # 반사광 크기에 따라 10~20픽셀 정도 옆의 이미지를 가져옴
    M = np.float32([[1, 0, 15], [0, 1, 15]]) 
    shift_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 3. Seamless Cloning (Poisson Blending)
    # OpenCV의 이 함수는 소스 영역의 '질감'만 가져오고 '색감'은 대상의 주변과 맞춥니다.
    # 회색/핑크 덩어리 현상을 막는 핵심 알고리즘입니다.
    
    # 마스크의 중심점 찾기
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img

    final = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2: continue # 너무 작은 노이즈 무시
        
        center = (x + w // 2, y + h // 2)
        
        # 각 마스크 영역에 대해 Seamless Clone 수행
        try:
            # MIXED_CLONE을 사용하면 원본 혈관의 흐름을 어느 정도 유지하며 합성됩니다.
            final = cv2.seamlessClone(shift_img, final, dilated_mask, center, cv2.MIXED_CLONE)
        except:
            # 이미지 경계 근처에서 에러 방지용
            continue

    # 4. (선택사항) 질감 보강을 위한 미세 노이즈 추가
    # 너무 매끈해서 가짜처럼 보일 경우를 대비해 아주 미세하게 질감을 입힙니다.
    noise = np.random.normal(0, 1.5, final.shape).astype(np.uint8)
    final = cv2.add(final, noise)

    return final


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
