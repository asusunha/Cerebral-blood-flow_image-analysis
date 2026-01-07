import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

class UltraPreciseVesselAligner:
    def __init__(self):
        """
        초기 ROI 사이즈는 None으로 설정하며, set_baseline에서 마우스로 결정합니다.
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

    def preprocess_vesselness(self, img_gray):
        """[핵심 알고리즘: Vesselness 필터]"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
        return v_map

    def set_baseline(self, img_path):
        """마우스 드래그로 ROI를 설정하고 기준 특징을 추출합니다."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")
        
        self.full_baseline_img = img
        self.baseline_name = os.path.splitext(os.path.basename(img_path))[0]

        # --- 마우스 ROI 선택 로직 추가 ---
        window_name = "Select ROI (Drag & Press ENTER or SPACE) / ESC to Cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 창 크기 조절 가능하게 설정
        
        # 마우스로 드래그하여 ROI 선택 (x, y, w, h 반환)
        roi = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        
        x, y, w, h = roi
        
        if w == 0 or h == 0:
            print("ROI 선택이 취소되었거나 잘못되었습니다. 다시 시도해주세요.")
            return None

        self.roi_offset = (x, y)
        self.baseline_roi_size = (w, h)
        
        print(f"선택된 ROI: x={x}, y={y}, w={w}, h={h}")

        # 선택된 ROI 영역 추출 및 전처리
        roi_color = img[y:y+h, x:x+w]
        self.baseline_roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(self.baseline_roi_gray)

        print(f"Baseline 설정 완료: {os.path.basename(img_path)}")
        return roi_color

    def align_and_crop(self, target_img_path, save_dir):
        if self.roi_offset is None:
            print("먼저 set_baseline을 호출하여 ROI를 설정해야 합니다.")
            return None

        target_img = cv2.imread(target_img_path)
        if target_img is None: return None
        
        target_name = os.path.basename(target_img_path)
        x, y = self.roi_offset
        w, h = self.baseline_roi_size

        # 1. 매칭을 위한 Target ROI 추출 및 전처리 (Baseline과 동일 위치)
        target_roi_gray = cv2.cvtColor(target_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
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
            print(f"ECC 정합 성공 (상관계수: {cc:.4f})")
        except cv2.error:
            print(f"ECC 실패. 초기 정합 결과 사용.")

        # 4. 전체 타겟 이미지를 워핑
        full_h, full_w = target_img.shape[:2]
        warped_full_img = cv2.warpAffine(
            target_img, 
            warp_matrix, 
            (full_w, full_h), 
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT # 외곽 블랙 최소화
        )

        # 5. 워핑된 전체 이미지에서 ROI 영역만큼 크롭
        aligned_roi = warped_full_img[y:y+h, x:x+w]

        # 저장 로직
        baseline_save_path = os.path.join(save_dir, f"Baseline_{self.baseline_name}.tiff")
        if not os.path.exists(baseline_save_path):
            cv2.imwrite(baseline_save_path, self.full_baseline_img[y:y+h, x:x+w])
            
        output_filename = f"Aligned_{os.path.splitext(target_name)[0]}.tiff"
        cv2.imwrite(os.path.join(save_dir, output_filename), aligned_roi)
        
        return aligned_roi

# --- 실행부 ---
if __name__ == "__main__":
    # 경로 설정 (사용자 경로에 맞게 수정)
    BASELINE_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00001'
    TARGET_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00060'
    
    BASELINE_NAME = 'Video_00056_00001_0s.tiff'
    TARGET_NAME = 'Video_00056_00060_0s.tiff'
    
    SAVE_DIR = os.path.join(TARGET_DIR, "Aligned_ROI_v5_drag")
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 1. 정렬 객체 생성 (사이즈 미리 지정 안 함)
    aligner = UltraPreciseVesselAligner()

    # 2. 기준 이미지 설정 (여기서 마우스 드래그 창이 뜹니다)
    baseline_full_path = os.path.join(BASELINE_DIR, BASELINE_NAME)
    aligner.set_baseline(baseline_full_path)
    
    # 3. 대상 이미지 정렬 및 크롭 저장
    target_full_path = os.path.join(TARGET_DIR, TARGET_NAME)
    result = aligner.align_and_crop(target_full_path, SAVE_DIR)