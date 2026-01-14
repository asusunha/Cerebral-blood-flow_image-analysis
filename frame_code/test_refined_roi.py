import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

def load_frames_from_dir(frame_dir, max_frames=20):
    """디렉토리에서 최대 20개의 프레임을 로드"""
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    frame_files = frame_files[:max_frames]
    frames = [cv2.imread(f) for f in frame_files]
    return frames, [os.path.basename(f) for f in frame_files]

class IntegratedVesselAligner:
    def __init__(self, abs_threshold=15.0):
        self.abs_threshold = abs_threshold
        self.roi_offset = None # (x, y)
        self.baseline_roi_size = None # (w, h)
        self.baseline_vesselness = None
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def get_refined_mask(self, img_bgr):
        """[FinalVesselSystem 로직] A-채널 기반으로 조직 영역 마스크 추출"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        blur = cv2.GaussianBlur(a_channel, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((25, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255 # 실패 시 전체 허용
        
        main_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_cnt)
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask

    def calculate_masked_clarity(self, img_bgr):
        """마스크 영역 내에서만 선명도 계산"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = self.get_refined_mask(img_bgr)
        
        # 라플라시안 계산
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 마스크 영역(흰색)의 라플라시안 값들만 추출
        mask_values = laplacian[mask > 0]
        
        if len(mask_values) == 0:
            return 0
        return mask_values.var()

    def preprocess_vesselness(self, img_gray):
        """ECC 정합용 Vesselness 맵 생성"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        return cv2.GaussianBlur(v_map, (3, 3), 0)

    def set_baseline(self, baseline_dir):
        frames, _ = load_frames_from_dir(baseline_dir, max_frames=1)
        if not frames: raise ValueError("Baseline 이미지가 없습니다.")
        
        # ROI 드래그 선택
        img = frames[0]
        window_name = "Select ROI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(window_name, img, False)
        cv2.destroyWindow(window_name)
        
        x, y, w, h = roi
        self.roi_offset = (x, y)
        self.baseline_roi_size = (w, h)
        
        # Baseline 정합 기준 맵 생성
        roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(roi_gray)
        print(f"✅ Baseline ROI 설정 완료 ({w}x{h})")

    def process_align(self, frame_dir, save_dir, prefix, max_frames=20):
        os.makedirs(save_dir, exist_ok=True)
        frames, filenames = load_frames_from_dir(frame_dir, max_frames=max_frames)
        
        x, y = self.roi_offset
        w, h = self.baseline_roi_size
        
        success_count = 0
        skip_count = 0

        print(f"\n🚀 [{prefix}] 필터링 및 정합 시작...")

        for i, frame in enumerate(frames):
            # 1. ROI 영역 자르기
            roi_bgr = frame[y:y+h, x:x+w]
            
            # 2. 혈관 타겟 선명도 계산
            score = self.calculate_masked_clarity(roi_bgr)
            
            # 3. 절대값 필터링
            if score < self.abs_threshold:
                print(f"  [-] Frame {i:02d} 제외 (혈관 선명도: {score:.2f})")
                skip_count += 1
                continue
            
            # 4. ECC 정합 진행
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            frame_v = self.preprocess_vesselness(roi_gray)
            
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    self.baseline_vesselness, frame_v, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, self.criteria
                )
            except: pass # 실패 시 변형 없이 진행
            
            # 5. 워핑 및 크롭 저장
            h_f, w_f = frame.shape[:2]
            warped = cv2.warpAffine(frame, warp_matrix, (w_f, h_f),
                                   flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT)
            
            save_name = f"{prefix}_{i:02d}.tiff"
            cv2.imwrite(os.path.join(save_dir, save_name), warped[y:y+h, x:x+w])
            success_count += 1
            print(f"  [+] Frame {i:02d} 저장 완료 (점수: {score:.2f})")

        return success_count, skip_count

# --- 메인 실행부 ---
if __name__ == "__main__":
    # abs_threshold: 이제 이 점수는 '혈관 영역 내'의 선명도입니다.
    # 배경 노이즈가 빠졌으므로 이전보다 점수가 더 순수하게 혈관 상태를 반영합니다.
    aligner = IntegratedVesselAligner(abs_threshold=20.0) 

    BASE_PATH = r"D:\VIDEO\M-24\Video_00056_00001_0to5s_tiff"
    TARGET_PATH = r"D:\VIDEO\M-24\Video_00056_00052_2m30sto2m35s_tiff"
    SAVE_ROOT = r"D:\VIDEO\M-24\ROI_window_ext_1_52_all"

    # 1. 기준 설정 (드래그)
    aligner.set_baseline(BASE_PATH)

    # 2. Baseline 처리 (20장 중 선별)
    b_ok, b_skip = aligner.process_align(BASE_PATH, os.path.join(SAVE_ROOT, "Baseline"), "base")

    # 3. Target 처리 (20장 중 선별)
    t_ok, t_skip = aligner.process_align(TARGET_PATH, os.path.join(SAVE_ROOT, "Target"), "target")

    print("\n" + "="*50)
    print(f"🏁 최종 결과 보고 (대상: 각 폴더 상위 20장)")
    print(f" - Baseline: {b_ok}장 성공 / {b_skip}장 제외")
    print(f" - Target: {t_ok}장 성공 / {t_skip}장 제외")
    print(f" - 저장경로: {SAVE_ROOT}")
    print("="*50)