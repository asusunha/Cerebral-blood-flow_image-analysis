'''
코드 상에서 ROI 좌표 직접 지정
ECC 기반 Aligned Image 추출
+ RANSAC 옵션 추가
+ 이미지 피라미드(Multi-scale) 정합 도입
'''

import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

class UltraPreciseVesselAligner:
    def __init__(self, pyramid_levels=3):
        """
        pyramid_levels: 이미지 피라미드 레벨 (기본값 3)
        """
        self.baseline_roi_size = None
        self.baseline_roi_gray = None
        self.baseline_roi_color = None
        self.baseline_vesselness = None
        self.roi_offset = None # (x, y)
        self.full_baseline_img = None
        self.baseline_name = ""
        
        # ECC 알고리즘 종료 조건
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
        
        # 이미지 피라미드 레벨
        self.pyramid_levels = pyramid_levels

    def preprocess_vesselness(self, img_gray):
        """[핵심 알고리즘: Vesselness 필터]"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
        return v_map

    def build_pyramid(self, img, levels):
        """이미지 피라미드 구축 (Gaussian Pyramid)"""
        pyramid = [img]
        for i in range(levels):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    def set_baseline(self, img_path, roi_coords):
        """
        ROI를 직접 지정하여 기준 특징을 추출합니다.
        
        Parameters:
        -----------
        img_path : str
            Baseline 이미지 경로
        roi_coords : tuple
            (x, y, width, height) 형태의 ROI 좌표
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")
        
        self.full_baseline_img = img
        self.baseline_name = os.path.splitext(os.path.basename(img_path))[0]

        # ROI 좌표 설정
        x, y, w, h = roi_coords
        
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI 크기가 잘못되었습니다: width={w}, height={h}")
        
        # 이미지 경계 체크
        img_h, img_w = img.shape[:2]
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            raise ValueError(f"ROI가 이미지 경계를 벗어났습니다. 이미지 크기: ({img_w}, {img_h}), ROI: ({x}, {y}, {w}, {h})")

        self.roi_offset = (x, y)
        self.baseline_roi_size = (w, h)
        
        print(f"✓ 설정된 ROI: x={x}, y={y}, w={w}, h={h}")

        # 선택된 ROI 영역 추출 및 전처리
        roi_color = img[y:y+h, x:x+w]
        self.baseline_roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(self.baseline_roi_gray)

        print(f"✓ Baseline 설정 완료: {os.path.basename(img_path)}")
        return roi_color

    def align_and_crop(self, target_img_path, save_dir):
        if self.roi_offset is None:
            print("먼저 set_baseline을 호출하여 ROI를 설정해야 합니다.")
            return None

        target_img = cv2.imread(target_img_path)
        if target_img is None: 
            print(f"✗ 타겟 이미지를 읽을 수 없습니다: {target_img_path}")
            return None
        
        target_name = os.path.basename(target_img_path)
        x, y = self.roi_offset
        w, h = self.baseline_roi_size

        # 1. 매칭을 위한 Target ROI 추출 및 전처리 (Baseline과 동일 위치)
        target_roi_gray = cv2.cvtColor(target_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        target_vesselness = self.preprocess_vesselness(target_roi_gray)

        # ===== 이미지 피라미드 구축 =====
        baseline_pyramid = self.build_pyramid(self.baseline_vesselness, self.pyramid_levels)
        target_pyramid = self.build_pyramid(target_vesselness, self.pyramid_levels)

        # ===== 초기 행렬 추정 (ORB + RANSAC) =====
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # 가장 작은 스케일(피라미드 최상단)에서 ORB 특징점 매칭
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(baseline_pyramid[-1], None)
        kp2, des2 = orb.detectAndCompute(target_pyramid[-1], None)
        
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # RANSAC 옵션 활성화
                M_init, inliers = cv2.estimateAffinePartial2D(
                    src_pts, 
                    dst_pts, 
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=2000,
                    confidence=0.99
                )
                
                if M_init is not None:
                    # 피라미드 최상단에서 구한 변환은 원본 스케일로 복원
                    scale_factor = 2 ** self.pyramid_levels
                    M_init[:, 2] *= scale_factor
                    warp_matrix = M_init.astype(np.float32)
                    
                    if inliers is not None:
                        inlier_count = np.sum(inliers)
                        print(f"  ORB+RANSAC: {len(matches)}개 매칭 중 {inlier_count}개 inlier")
                else:
                    print(f"  ORB+RANSAC 실패 (identity로 초기화)")
        else:
            print(f"  ORB 특징점 부족 (identity로 초기화)")

        # ===== Multi-scale ECC 정합 (coarse-to-fine) =====
        for level in range(self.pyramid_levels, -1, -1):
            baseline_level = baseline_pyramid[level]
            target_level = target_pyramid[level]
            
            try:
                (cc, warp_matrix) = cv2.findTransformECC(
                    baseline_level, 
                    target_level, 
                    warp_matrix, 
                    cv2.MOTION_EUCLIDEAN, 
                    self.criteria,
                    None,
                    5
                )
                print(f"  Level {level} ECC 성공 (상관계수: {cc:.4f})")
                
                # 다음 레벨로 올라갈 때 변환 행렬 스케일 조정
                if level > 0:
                    warp_matrix[:, 2] *= 2
                    
            except cv2.error as e:
                print(f"  Level {level} ECC 실패: {e}")
                if level > 0:
                    warp_matrix[:, 2] *= 2
                continue

        # 4. 전체 타겟 이미지를 워핑
        full_h, full_w = target_img.shape[:2]
        warped_full_img = cv2.warpAffine(
            target_img, 
            warp_matrix, 
            (full_w, full_h), 
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        # 5. 워핑된 전체 이미지에서 ROI 영역만큼 크롭
        aligned_roi = warped_full_img[y:y+h, x:x+w]

        # 저장 로직
        baseline_save_path = os.path.join(save_dir, f"Baseline_{self.baseline_name}.tiff")
        if not os.path.exists(baseline_save_path):
            cv2.imwrite(baseline_save_path, self.full_baseline_img[y:y+h, x:x+w])
            print(f"✓ Baseline 이미지 저장: {baseline_save_path}")
            
        output_filename = f"Aligned_{os.path.splitext(target_name)[0]}.tiff"
        output_path = os.path.join(save_dir, output_filename)
        cv2.imwrite(output_path, aligned_roi)
        print(f"✓ Aligned 이미지 저장: {output_path}")
        
        return aligned_roi

# --- 실행부 ---
if __name__ == "__main__":
    # 경로 설정
    BASELINE_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00001'
    TARGET_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00060'
    
    BASELINE_NAME = 'Video_00056_00001_0s.tiff'
    TARGET_NAME = 'Video_00056_00060_0s.tiff'
    
    SAVE_DIR = os.path.join(TARGET_DIR, "Aligned_ROI_v5_drag")
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)

    # ===== ROI 좌표 직접 지정 (x, y, width, height) =====
    # 예시: 이미지의 중앙 1800x1600 영역
    # 실제 사용 시 원하는 혈관 영역 좌표로 변경하세요
    ROI_COORDS = (1356, 308, 1338, 1195)  # (x, y, width, height)
    
    # 또는 이미지 크기를 먼저 확인하고 중앙 기준으로 자동 계산:
    # temp_img = cv2.imread(os.path.join(BASELINE_DIR, BASELINE_NAME))
    # img_h, img_w = temp_img.shape[:2]
    # roi_w, roi_h = 1800, 1600
    # ROI_COORDS = ((img_w - roi_w) // 2, (img_h - roi_h) // 2, roi_w, roi_h)

    # 1. 정렬 객체 생성
    aligner = UltraPreciseVesselAligner(pyramid_levels=3)

    # 2. 기준 이미지 설정 (ROI 좌표 직접 전달)
    baseline_full_path = os.path.join(BASELINE_DIR, BASELINE_NAME)
    aligner.set_baseline(baseline_full_path, ROI_COORDS)
    
    # 3. 대상 이미지 정렬 및 크롭 저장
    target_full_path = os.path.join(TARGET_DIR, TARGET_NAME)
    result = aligner.align_and_crop(target_full_path, SAVE_DIR)
    
    if result is not None:
        print("\n" + "="*60)
        print("정렬 완료! 결과 확인 경로:")
        print(f"  → {SAVE_DIR}")
        print("="*60)