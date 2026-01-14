import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

def load_frames_from_dir(frame_dir, max_frames=10):
    """디렉토리에서 상위 N개의 TIFF 프레임들을 로드"""
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    frame_files = frame_files[:max_frames]  # 상위 N개만
    frames = [cv2.imread(f) for f in frame_files]
    filenames = [os.path.basename(f) for f in frame_files]
    return frames, filenames


class ROIBasedVesselAligner:
    def __init__(self, abs_threshold=20.0):
        self.abs_threshold = abs_threshold
        self.roi_offset = None  # (x, y)
        self.roi_size = None  # (w, h)
        self.baseline_vesselness = None
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def get_refined_mask(self, img_bgr):
        """A-Channel 기반 혈관 영역 마스크 생성"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        blur = cv2.GaussianBlur(a_channel, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
        kernel = np.ones((25, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
        
        main_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_cnt)
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask

    def calculate_masked_clarity(self, img_bgr):
        """Sobel Gradient 기반 혈관 영역 선명도 측정"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = self.get_refined_mask(img_bgr)
        
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        mag = cv2.magnitude(dx, dy)
        
        vessel_area_mag = mag[mask > 0]
        if len(vessel_area_mag) == 0:
            return 0
        
        # 상위 10% 엣지값 평균
        score = np.mean(np.sort(vessel_area_mag)[-int(len(vessel_area_mag)*0.1):])
        return score

    def preprocess_vesselness(self, img_gray):
        """ECC 정합용 Vesselness 맵 생성"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        return cv2.GaussianBlur(v_map, (3, 3), 0)

    def find_first_clear_frame(self, frames):
        """선명도 통과한 첫 프레임 찾기"""
        for i, frame in enumerate(frames):
            score = self.calculate_masked_clarity(frame)
            if score >= self.abs_threshold:
                print(f"✅ 첫 통과 프레임: {i}번 (선명도: {score:.2f})")
                return frame, i
        raise ValueError("선명도 통과한 프레임이 없습니다!")

    def set_baseline_roi(self, baseline_dir, max_frames=10):
        """Baseline 디렉토리에서 첫 통과 프레임으로 ROI 설정"""
        frames, _ = load_frames_from_dir(baseline_dir, max_frames=max_frames)
        if not frames:
            raise ValueError("Baseline 이미지가 없습니다.")
        
        # 선명도 통과한 첫 프레임 찾기
        first_clear_frame, idx = self.find_first_clear_frame(frames)
        
        # ROI 드래그 선택
        window_name = "Select ROI on First Clear Frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("💡 드래그로 ROI를 선택한 후 'ENTER' 또는 'SPACE'를 누르세요.")
        roi = cv2.selectROI(window_name, first_clear_frame, False)
        cv2.destroyWindow(window_name)
        
        x, y, w, h = roi
        if w == 0 or h == 0:
            raise ValueError("ROI가 선택되지 않았습니다.")
        
        self.roi_offset = (x, y)
        self.roi_size = (w, h)
        
        # Baseline ROI 기준 Vesselness 맵 생성
        roi_gray = cv2.cvtColor(first_clear_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(roi_gray)
        
        print(f"✅ Baseline ROI 설정 완료: x={x}, y={y}, w={w}, h={h}")
        return first_clear_frame, idx

    def process_frames(self, frame_dir, save_dir, prefix, max_frames=10, is_baseline=False, baseline_first_idx=None):
        """프레임 필터링 및 정합 처리"""
        os.makedirs(save_dir, exist_ok=True)
        frames, filenames = load_frames_from_dir(frame_dir, max_frames=max_frames)
        
        if self.roi_offset is None:
            raise ValueError("먼저 set_baseline_roi()를 호출하세요.")
        
        x, y = self.roi_offset
        w, h = self.roi_size
        
        success_count = 0
        skip_count = 0
        
        print(f"\n🚀 [{prefix}] 필터링 및 정합 시작...")
        
        for i, frame in enumerate(frames):
            # Baseline의 첫 통과 프레임은 정렬 없이 저장
            if is_baseline and i == baseline_first_idx:
                roi_bgr = frame[y:y+h, x:x+w]
                save_name = f"{prefix}_{i:02d}.tiff"
                cv2.imwrite(os.path.join(save_dir, save_name), roi_bgr)
                success_count += 1
                print(f"  [+] Frame {i:02d} 저장 완료 (기준 프레임)")
                continue
            
            # ROI 영역 자르기
            roi_bgr = frame[y:y+h, x:x+w]
            
            # 선명도 계산 (유효 영역만)
            score = self.calculate_masked_clarity(roi_bgr)
            
            # 절대값 필터링
            if score < self.abs_threshold:
                print(f"  [-] Frame {i:02d} 제외 (선명도: {score:.2f})")
                skip_count += 1
                continue
            
            # ECC 정합 수행
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            frame_v = self.preprocess_vesselness(roi_gray)
            
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    self.baseline_vesselness, frame_v, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, self.criteria
                )
                
                # ROI 워핑 (충분한 여유 공간)
                pad = 50
                warped_roi = cv2.warpAffine(
                    roi_bgr, warp_matrix, (w + 2*pad, h + 2*pad),
                    flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT
                )
                cropped = warped_roi[pad:pad+h, pad:pad+w]
                
                save_name = f"{prefix}_{i:02d}.tiff"
                cv2.imwrite(os.path.join(save_dir, save_name), cropped)
                success_count += 1
                print(f"  [+] Frame {i:02d} 저장 완료 (선명도: {score:.2f})")
                
            except cv2.error as e:
                print(f"  [-] Frame {i:02d} ECC 정합 실패 (선명도: {score:.2f})")
                skip_count += 1
                continue
        
        return success_count, skip_count


# --- 메인 실행부 ---
if __name__ == "__main__":
    aligner = ROIBasedVesselAligner(abs_threshold=20.0)
    
    BASE_PATH = r"D:\VIDEO\M-24\Video_00056_00001_0to5s_tiff"
    TARGET_PATH = r"D:\VIDEO\M-24\Video_00056_00052_2m30sto2m35s_tiff"
    SAVE_ROOT = r"D:\VIDEO\M-24\ROI_window_final"
    
    print("="*60)
    print("🚀 ROI 기반 혈관 이미지 정렬 시작 (상위 10개 프레임)")
    print("="*60)
    
    # 1. Baseline 첫 통과 프레임에서 ROI 설정
    print("\n[STEP 1] Baseline ROI 설정")
    first_frame, first_idx = aligner.set_baseline_roi(BASE_PATH, max_frames=10)
    
    # 2. Baseline 전체 처리
    print("\n[STEP 2] Baseline 프레임 처리")
    b_ok, b_skip = aligner.process_frames(
        BASE_PATH, 
        os.path.join(SAVE_ROOT, "Baseline"), 
        "base",
        max_frames=10,
        is_baseline=True,
        baseline_first_idx=first_idx
    )
    
    # 3. Target 전체 처리
    print("\n[STEP 3] Target 프레임 처리")
    t_ok, t_skip = aligner.process_frames(
        TARGET_PATH,
        os.path.join(SAVE_ROOT, "Target"),
        "target",
        max_frames=10,
        is_baseline=False
    )
    
    print("\n" + "="*60)
    print(f"🏁 최종 결과 보고 (대상: 각 폴더 상위 10개 프레임)")
    print(f" - Baseline: {b_ok}장 성공 / {b_skip}장 제외")
    print(f" - Target: {t_ok}장 성공 / {t_skip}장 제외")
    print(f" - 저장경로: {SAVE_ROOT}")
    print("="*60)