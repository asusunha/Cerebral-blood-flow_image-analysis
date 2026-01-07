'''
이미지 자체의 혈관 네트워크 비교 (SSIM)
'''

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def verify_alignment_visual(baseline_path, aligned_path):
    # 1. 이미지 로드
    img_base = cv2.imread(baseline_path)
    img_aligned = cv2.imread(aligned_path)

    if img_base is None or img_aligned is None:
        print("이미지 경로를 확인해주세요.")
        return

    # 2. 유효 영역 마스크 (블랙 영역 제외)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    mask = (gray_aligned > 0).astype(np.uint8) * 255

    # 3. 유사도 수치 계산 (유효 영역 내 SSIM)
    score, diff = ssim(gray_base, gray_aligned, full=True)
    valid_score = diff[mask > 0].mean()

    # 4. 육안 확인을 위한 컬러 오버레이 생성 (Anaglyph 방식)
    # 기준(Base) 이미지를 Red 채널에, 정렬(Aligned) 이미지를 Cyan(Green+Blue) 채널에 배치
    # 정합이 완벽하면 겹친 부분이 '흰색(무채색)'으로 보이고, 어긋나면 색 분리가 일어남
    h, w = gray_base.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    overlay[:, :, 2] = gray_base     # Red 채널: Baseline
    overlay[:, :, 1] = gray_aligned  # Green 채널: Aligned
    overlay[:, :, 0] = gray_aligned  # Blue 채널: Aligned

    # 5. 투명도 조절 합성 (Alpha Blending) - 50:50 비율
    alpha_blended = cv2.addWeighted(img_base, 0.5, img_aligned, 0.5, 0)

    # 6. 결과 출력
    print(f"\n[정합 위치 확인 결과]")
    print("-" * 40)
    print(f"📍 위치 정합도 (SSIM): {valid_score:.4f}")
    print(f"   (1.0에 가까울수록 위치가 정확히 포개진 상태)")
    print("-" * 40)
    print("💡 가이드:")
    print(" - 컬러 오버레이: 흰색/회색으로 보이면 일치, 빨강/파랑 잔상이 보이면 어긋남")
    print(" - 투명도 합성: 두 이미지가 50% 투명도로 겹쳐 보임")

    # 결과창 띄우기
    cv2.imshow("1. Color Overlay (Red-Cyan)", overlay)
    cv2.imshow("2. Alpha Blended (50-50)", alpha_blended)
    
    # 결과 저장 (선택 사항)
    save_path = os.path.join(os.path.dirname(aligned_path), "Verification_Overlay.jpg")
    cv2.imwrite(save_path, overlay)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ROOT_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030\Aligned_ROI_v2'
    
    base_img = os.path.join(ROOT_DIR, 'Baseline_Video_00056_00001_0s.tiff')
    align_img = os.path.join(ROOT_DIR, 'Aligned_Video_00056_00030_0s.tiff')

    verify_alignment_visual(base_img, align_img)