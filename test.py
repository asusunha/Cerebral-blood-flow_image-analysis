import cv2
import numpy as np
import os
from skimage.morphology import skeletonize


def binarize_vessel(gray):
    """
    혈관 이진화 (조명 변화에 강한 방식)
    """
    # CLAHE로 대비 정규화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    # Otsu Threshold
    _, binary = cv2.threshold(
        norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 혈관이 어두운 경우 반전
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    return binary


def skeletonize_binary(binary):
    """
    Binary → Skeleton (1px)
    """
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    return (skeleton.astype(np.uint8)) * 255


def dilate_skeleton(skel, radius=2):
    """
    Skeleton에 허용 반경 부여 (tolerance)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(skel, kernel, iterations=1)


def dice_coefficient(mask1, mask2):
    """
    Dice 계수
    """
    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    size_sum = m1.sum() + m2.sum()

    if size_sum == 0:
        return 0.0

    return 2.0 * intersection / size_sum


def verify_alignment_skeleton(baseline_path, aligned_path):
    # 1. 이미지 로드
    img_base = cv2.imread(baseline_path)
    img_aligned = cv2.imread(aligned_path)

    if img_base is None or img_aligned is None:
        print("❌ 이미지 로드 실패")
        return

    # 2. Grayscale 변환
    gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)

    # 3. 혈관 이진화
    bin_base = binarize_vessel(gray_base)
    bin_aligned = binarize_vessel(gray_aligned)

    # 4. Skeletonization (중요: 여기 누락되면 안 됨)
    skel_base = skeletonize_binary(bin_base)
    skel_aligned = skeletonize_binary(bin_aligned)

    # 5. Tolerance 적용 (±2px)
    skel_base_tol = dilate_skeleton(skel_base, radius=2)
    skel_aligned_tol = dilate_skeleton(skel_aligned, radius=2)

    # 6. Skeleton Dice 계산
    dice = dice_coefficient(skel_base_tol, skel_aligned_tol)

    # 7. 시각화 (Skeleton Overlay)
    overlay = img_aligned.copy()
    overlay[skel_base > 0] = [0, 0, 255]       # Baseline Skeleton → Red
    overlay[skel_aligned > 0] = [255, 255, 0] # Aligned Skeleton → Cyan

    # 8. 결과 출력
    print("\n[Skeleton 기반 혈관 정합 결과]")
    print("-" * 45)
    print(f"🧠 Skeleton Dice (±2px tolerance): {dice:.4f}")
    print("  (1.0 = 혈관 네트워크 완전 겹침)")
    print("-" * 45)
    print("📌 해석 가이드:")
    print(" - 0.85 이상 : 매우 우수한 정합")
    print(" - 0.75~0.85 : 실사용 충분")
    print(" - 0.70 미만 : 정합 문제 가능")

    # 9. 결과 저장
    save_dir = os.path.dirname(aligned_path)
    cv2.imwrite(os.path.join(save_dir, "Skeleton_Overlay.png"), overlay)
    cv2.imwrite(os.path.join(save_dir, "Skeleton_Base.png"), skel_base)
    cv2.imwrite(os.path.join(save_dir, "Skeleton_Aligned.png"), skel_aligned)
    cv2.imwrite(os.path.join(save_dir, "Skeleton_Base_Tolerance.png"), skel_base_tol)
    cv2.imwrite(os.path.join(save_dir, "Skeleton_Aligned_Tolerance.png"), skel_aligned_tol)

    # 10. 표시
    cv2.imshow("Skeleton Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ROOT_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030\Aligned_ROI_v4'

    base_img = os.path.join(
        ROOT_DIR, "Baseline_Video_00056_00001_0s.tiff"
    )
    align_img = os.path.join(
        ROOT_DIR, "Aligned_Video_00056_00030_0s.tiff"
    )

    verify_alignment_skeleton(base_img, align_img)
