import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

def load_frames_from_dir(frame_dir, max_frames=20):
    """디렉토리에서 최대 20개의 TIFF 프레임들을 로드"""
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    # 딱 20장만 슬라이싱
    frame_files = frame_files[:max_frames]
    
    frames = [cv2.imread(f) for f in frame_files]
    return frames, [os.path.basename(f) for f in frame_files]


class UltraPreciseVesselAligner:
    def __init__(self, abs_threshold=15.0):
        """
        abs_threshold: 선명도 절대 기준점 (이 값보다 낮으면 흔들린 것으로 간주)
        """
        self.baseline_roi_size = None
        self.baseline_roi_gray = None
        self.baseline_vesselness = None
        self.roi_offset = None
        self.baseline_frames = []
        self.baseline_filenames = []
        self.abs_threshold = abs_threshold
        
        # ECC 알고리즘 종료 조건
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def calculate_clarity(self, img_gray):
        """이미지의 선명도(절대값) 계산"""
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

    def preprocess_vesselness(self, img_gray):
        """[핵심 알고리즘: Vesselness 필터]"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
        return v_map

    def set_baseline_from_frames(self, baseline_frame_dir, max_frames=20): 
        frames, filenames = load_frames_from_dir(baseline_frame_dir, max_frames=max_frames) 
        if len(frames) == 0: 
            raise ValueError("Baseline 디렉토리가 비어있습니다.") 

        self.baseline_frames = frames
        self.baseline_filenames = filenames
        first_frame = frames[0]
        
        window_name = "Select ROI (First Baseline Frame)" 
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) 
        
        print("💡 ROI를 드래그한 후 ENTER를 누르세요.")
        roi = cv2.selectROI(window_name, first_frame, fromCenter=False) 
        cv2.destroyWindow(window_name) 
        
        x, y, w, h = roi 
        if w == 0 or h == 0:
            print("❌ ROI 미선택.")
            return

        self.roi_offset = (x, y) 
        self.baseline_roi_size = (w, h)
        
        baseline_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.baseline_roi_gray = baseline_gray[y:y+h, x:x+w]
        self.baseline_vesselness = self.preprocess_vesselness(self.baseline_roi_gray) 
        print(f"✅ Baseline 설정 완료: x={x}, y={y}, w={w}, h={h}")

    def process_baseline_frames_with_ecc(self, save_root_dir):
        if not self.baseline_frames: return
        
        x, y = self.roi_offset
        w, h = self.baseline_roi_size
        baseline_dir = os.path.join(save_root_dir, "Baseline_Filtered")
        os.makedirs(baseline_dir, exist_ok=True)
        
        skip_count = 0
        success_count = 0

        print(f"\n📁 Baseline 필터링 및 정합 시작 (기준: {self.abs_threshold})")
        
        for i, frame in enumerate(self.baseline_frames):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = frame_gray[y:y+h, x:x+w]
            
            # 1. 선명도 체크
            score = self.calculate_clarity(roi_gray)
            if score < self.abs_threshold:
                print(f"  [-] Frame {i:02d} 제외 (점수: {score:.2f})")
                skip_count += 1
                continue
            
            # 2. 정합 진행
            frame_v = self.preprocess_vesselness(roi_gray)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    self.baseline_vesselness, frame_v, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, self.criteria
                )
            except: pass
            
            h_full, w_full = frame.shape[:2]
            warped = cv2.warpAffine(frame, warp_matrix, (w_full, h_full), 
                                   flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT)
            
            cv2.imwrite(os.path.join(baseline_dir, f"base_{i:02d}.tiff"), warped[y:y+h, x:x+w])
            success_count += 1
        
        print(f"✅ Baseline 완료: {success_count}장 저장 / {skip_count}장 제외")
        return skip_count

    def align_window_frames_with_ecc(self, target_frame_dir, save_root_dir, max_frames=20):
        frames, filenames = load_frames_from_dir(target_frame_dir, max_frames=max_frames)
        if not frames: return

        x, y = self.roi_offset
        w, h = self.baseline_roi_size
        aligned_dir = os.path.join(save_root_dir, f"Target_Filtered")
        os.makedirs(aligned_dir, exist_ok=True)

        skip_count = 0
        success_count = 0

        print(f"\n📁 Target 필터링 및 정합 시작 (기준: {self.abs_threshold})")
        
        for i, frame in enumerate(frames):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = frame_gray[y:y+h, x:x+w]
            
            score = self.calculate_clarity(roi_gray)
            if score < self.abs_threshold:
                print(f"  [-] Frame {i:02d} 제외 (점수: {score:.2f})")
                skip_count += 1
                continue
                
            frame_v = self.preprocess_vesselness(roi_gray)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    self.baseline_vesselness, frame_v, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, self.criteria
                )
            except: pass
            
            h_full, w_full = frame.shape[:2]
            warped = cv2.warpAffine(frame, warp_matrix, (w_full, h_full),
                                   flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT)
            
            cv2.imwrite(os.path.join(aligned_dir, f"target_{i:02d}.tiff"), warped[y:y+h, x:x+w])
            success_count += 1

        print(f"✅ Target 완료: {success_count}장 저장 / {skip_count}장 제외")
        return skip_count

# --- 실행부 ---
if __name__ == "__main__":
    # 30이 너무 높다면 10~15 정도로 시작해보세요.
    aligner = UltraPreciseVesselAligner(abs_threshold=15.0)

    BASELINE_FRAMES = r"D:\VIDEO\M-24\Video_00056_00001_0to5s_tiff"
    TARGET_FRAMES   = r"D:\VIDEO\M-24\Video_00056_00052_2m30sto2m35s_tiff"
    SAVE_ROOT_DIR   = r"D:\VIDEO\M-24\ROI_window_ext_1_52_all"

    # 처리할 최대 프레임 고정
    MAX_COUNT = 20

    print("=" * 60)
    print(f"🚀 상위 {MAX_COUNT}장 대상 선명도 필터링 및 ECC 정합")
    print("=" * 60)

    aligner.set_baseline_from_frames(BASELINE_FRAMES, max_frames=MAX_COUNT)
    b_skip = aligner.process_baseline_frames_with_ecc(SAVE_ROOT_DIR)
    t_skip = aligner.align_window_frames_with_ecc(TARGET_FRAMES, SAVE_ROOT_DIR, max_frames=MAX_COUNT)
    
    print("\n" + "=" * 60)
    print(f"🏁 최종 리포트")
    print(f" - Baseline 제외된 프레임: {b_skip}장")
    print(f" - Target 제외된 프레임: {t_skip}장")
    print(f" - 결과물 저장 위치: {SAVE_ROOT_DIR}")
    print("=" * 60)