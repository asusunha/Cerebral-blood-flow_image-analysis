import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def verify_alignment_full_frame(baseline_path, aligned_path):
    # 1. 이미지 로드
    img_base = cv2.imread(baseline_path)
    img_aligned = cv2.imread(aligned_path)

    if img_base is None or img_aligned is None:
        print("이미지 경로를 확인해주세요.")
        return

    # 2. 그레이스케일 변환 (SSIM 계산용)
    gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)

    # 3. 전체 영역 SSIM 계산 (마스크 없이 전체 비교)
    # score: 전체 이미지에 대한 평균 SSIM (0~1)
    # diff: 픽셀별 차이를 나타내는 이미지
    score, diff = ssim(gray_base, gray_aligned, full=True)

    # 4. 육안 확인용 컬러 오버레이 (Anaglyph)
    # Baseline을 Red에, Aligned를 Cyan(Green+Blue)에 배치
    h, w = gray_base.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 2] = gray_base     # Red: Baseline
    overlay[:, :, 1] = gray_aligned  # Green: Aligned
    overlay[:, :, 0] = gray_aligned  # Blue: Aligned

    # 5. 투명도 합성 (Alpha Blending)
    alpha_blended = cv2.addWeighted(img_base, 0.5, img_aligned, 0.5, 0)

    # 6. 결과 출력
    print(f"\n[전체 영역 정합도 확인 결과]")
    print("-" * 45)
    print(f"📍 전체 SSIM 지수: {score:.4f}")
    print(f"   (1.0에 가까울수록 전체 구조가 완벽히 일치)")
    print("-" * 45)
    print("💡 분석 가이드:")
    print(" - 수치가 낮을 경우: 혈관 어긋남뿐만 아니라 외곽 블랙 영역 차이도 반영됨")
    print(" - 시각적 확인: Overlay 이미지에서 색 번짐이 적을수록 정렬이 잘된 것")
    
    # 결과 저장
    save_dir = os.path.dirname(aligned_path)
    cv2.imwrite(os.path.join(save_dir, "Full_Verification_Overlay.jpg"), overlay)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 경로 설정
    ROOT_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030\Aligned_ROI_v5_drag'
    
    base_img = os.path.join(ROOT_DIR, 'Baseline_Video_00056_00001_0s.tiff')
    align_img = os.path.join(ROOT_DIR, 'Aligned_Video_00056_00030_0s.tiff')

    verify_alignment_full_frame(base_img, align_img)