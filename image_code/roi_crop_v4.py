'''
# 입력
Baseline Image: Video_00056_00001_0s
Target Image: Video_00056_00030_0s

# 출력
Aligned_ROI_v2 폴더
- crop된 Baseline Image
- crop된 Aligned Image

# 결과
정확도 좋으나, 블랙으로 표시되는 픽셀 발생
'''

import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

class UltraPreciseVesselAligner:
    def __init__(self, baseline_roi_size=(1648, 1424)):
        """
        baseline_roi_size: 기준 이미지에서 추출하여 정렬 기준으로 삼을 ROI 크기 (width, height)
        """
        self.baseline_roi_size = baseline_roi_size
        self.baseline_roi_gray = None
        self.baseline_roi_color = None
        self.baseline_vesselness = None  # Vesselness 맵 저장
        self.roi_offset = None
        
        # ECC 알고리즘 종료 조건 (반복 횟수 증가 및 정밀도 설정)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def preprocess_vesselness(self, img_gray):
        """
        [핵심 알고리즘: Vesselness 필터]
        헤시안 행렬의 고윳값을 분석하여 혈관(관 형태) 구조만 추출합니다.
        반사광(Glare)과 배경 노이즈를 0에 가깝게 억제하여 ECC 정합의 정확도를 극대화합니다.
        """
        # Sato 필터 적용 (sigmas: 감지할 혈관 굵기 범위, 보통 1~5 사이가 적당함)
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        
        # 0~255 범위로 정규화 (ECC 알고리즘 입력용)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        
        # 가우시안 블러로 미세 노이즈 평탄화
        v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
        return v_map

    def set_baseline(self, img_path):
        """기준 프레임을 설정하고 Vesselness 맵을 미리 계산합니다."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")
        
        self.full_baseline_img = img
        h, w = img.shape[:2]
        roi_w, roi_h = self.baseline_roi_size
        
        # 중앙 기준 ROI 좌표 계산
        x1 = max(0, (w // 2) - (roi_w // 2))
        y1 = max(0, (h // 2) - (roi_h // 2))
        self.roi_offset = (x1, y1)

        # 특징 추출은 ROI 영역에서 수행 (정밀도 유지)
        roi_color = img[y1:y1+roi_h, x1:x1+roi_w]
        self.baseline_roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(self.baseline_roi_gray)
        self.baseline_name = os.path.splitext(os.path.basename(img_path))[0]

        print(f" Baseline 설정 완료: {os.path.basename(img_path)}")
        return roi_color


    def align_and_crop(self, target_img_path, save_dir):
        target_img = cv2.imread(target_img_path)
        if target_img is None: return None
        
        target_name = os.path.basename(target_img_path)
        x1, y1 = self.roi_offset
        roi_w, roi_h = self.baseline_roi_size

        # 1. 매칭을 위한 Target ROI 추출 및 전처리
        target_roi_gray = cv2.cvtColor(target_img[y1:y1+roi_h, x1:x1+roi_w], cv2.COLOR_BGR2GRAY)
        target_vesselness = self.preprocess_vesselness(target_roi_gray)

        # 2. 초기 행렬 추정 (ORB)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(self.baseline_vesselness, None)
        kp2, des2 = orb.detectAndCompute(target_vesselness, None)
        
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                M_init, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M_init is not None:
                    warp_matrix = M_init.astype(np.float32)

        # 3. ECC 정밀 정합
        try:
            (cc, warp_matrix) = cv2.findTransformECC(
                self.baseline_vesselness, 
                target_vesselness, 
                warp_matrix, 
                cv2.MOTION_EUCLIDEAN, 
                self.criteria,
                None,
                5
            )
            print(f" ECC 정합 성공 (상관계수: {cc:.4f})")
        except cv2.error:
            print(f" ECC 실패. ORB 결과 사용.")

        # [핵심 변경 부분]
        # 4. 전체 타겟 이미지를 워핑 (ROI 조각이 아니라 전체를 움직임)
        # WARP_INVERSE_MAP 플래그는 findTransformECC 결과에 따라 결정 (보통 필요)
        full_h, full_w = target_img.shape[:2]
        warped_full_img = cv2.warpAffine(
            target_img, 
            warp_matrix, 
            (full_w, full_h), 
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT # 혹은 BORDER_REPLICATE로 외곽 블랙 최소화
        )

        # 5. 워핑된 전체 이미지에서 ROI 영역만큼 크롭
        aligned_roi = warped_full_img[y1:y1+roi_h, x1:x1+roi_w]

        # 저장 로직
        baseline_save_path = os.path.join(save_dir, f"Baseline_{self.baseline_name}.tiff")
        if not os.path.exists(baseline_save_path):
            cv2.imwrite(baseline_save_path, self.full_baseline_img[y1:y1+roi_h, x1:x1+roi_w])
            
        output_filename = f"Aligned_{os.path.splitext(target_name)[0]}.tiff"
        cv2.imwrite(os.path.join(save_dir, output_filename), aligned_roi)
        
        return aligned_roi

# --- 실행부 ---
if __name__ == "__main__":
    # 1. 경로 설정
    BASELINE_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00001'
    TARGET_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030'
    
    BASELINE_NAME = 'Video_00056_00001_0s.tiff'
    TARGET_NAME = 'Video_00056_00030_0s.tiff'
    
    SAVE_DIR = os.path.join(TARGET_DIR, "Aligned_ROI_v4")
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 2. 정렬 객체 생성 (사용자가 지정한 ROI 크기 적용)
    # aligner = UltraPreciseVesselAligner(baseline_roi_size=(1648, 1424))
    aligner = UltraPreciseVesselAligner(baseline_roi_size=(1848, 1624))

    # 3. 기준 이미지 설정 및 특징점 추출
    baseline_full_path = os.path.join(BASELINE_DIR, BASELINE_NAME)
    aligner.set_baseline(baseline_full_path)
    
    # 4. 대상 이미지 정렬 및 크롭 저장
    target_full_path = os.path.join(TARGET_DIR, TARGET_NAME)
    result = aligner.align_and_crop(target_full_path, SAVE_DIR)
        