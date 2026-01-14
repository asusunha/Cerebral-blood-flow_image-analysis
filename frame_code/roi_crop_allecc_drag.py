'''
drag를 통한 ROI 설정 로직 추가
ECC 기반 Aligned Image 추출
모든 프레임 개별 ECC 정합 방식
'''

import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

def load_frames_from_dir(frame_dir, max_frames=None):
    """디렉토리에서 TIFF 프레임들을 로드"""
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    frames = [cv2.imread(f) for f in frame_files]
    return frames


class UltraPreciseVesselAligner:
    def __init__(self):
        """
        초기 ROI 사이즈는 None으로 설정하며, set_baseline에서 마우스로 결정합니다.
        """
        self.baseline_roi_size = None
        self.baseline_roi_gray = None
        self.baseline_vesselness = None
        self.roi_offset = None # (x, y)
        self.full_baseline_img = None
        self.baseline_name = ""
        self.baseline_frames = []  # 전체 baseline 프레임 저장
        
        # ECC 알고리즘 종료 조건
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def preprocess_vesselness(self, img_gray):
        """[핵심 알고리즘: Vesselness 필터]"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
        return v_map

    def set_baseline_from_frames(self, baseline_frame_dir, max_frames=10): 
        """첫 번째 프레임을 baseline으로 설정"""
        frames = load_frames_from_dir(baseline_frame_dir, max_frames=max_frames) 
        if len(frames) == 0: 
            raise ValueError("Baseline frame directory is empty") 

        self.baseline_frames = frames  # 전체 프레임 저장
        
        # 첫 번째 프레임 선택 (인덱스 0)
        first_frame = frames[0]
        print(f"🔍 Baseline 첫 번째 프레임 사용 (총 {len(frames)}장 로드)")
        
        self.full_baseline_img = first_frame
        self.baseline_name = os.path.basename(baseline_frame_dir)

        # ROI 선택 창 설정
        window_name = "Select ROI (First Baseline Frame)" 
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) 
        
        print("💡 드래그로 ROI를 선택한 후 'ENTER' 또는 'SPACE'를 누르세요.")
        print("💡 선택을 취소하려면 'C'를 누르세요.")
        
        roi = cv2.selectROI(window_name, first_frame, fromCenter=False) 
        cv2.destroyWindow(window_name) 
        
        x, y, w, h = roi 
        
        if w == 0 or h == 0:
            print("❌ ROI가 선택되지 않았습니다. 프로그램을 종료합니다.")
            return

        self.roi_offset = (x, y) 
        self.baseline_roi_size = (w, h)
        
        # Baseline ROI 이미지 저장
        baseline_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.baseline_roi_gray = baseline_gray[y:y+h, x:x+w]
        self.baseline_vesselness = self.preprocess_vesselness(self.baseline_roi_gray) 

        print(f"✅ Baseline 설정 완료: x={x}, y={y}, w={w}, h={h}")

    def process_baseline_frames_with_ecc(self, save_root_dir):
        """Baseline 전체 프레임을 개별 ECC 정합 후 crop하여 저장"""
        if self.roi_offset is None or len(self.baseline_frames) == 0:
            print("❌ Baseline이 설정되지 않았습니다.")
            return
        
        x, y = self.roi_offset
        w, h = self.baseline_roi_size
        
        baseline_dir = os.path.join(save_root_dir, "Baseline_window_0to5s_0")
        os.makedirs(baseline_dir, exist_ok=True)
        
        print(f"\n📁 Baseline {len(self.baseline_frames)}개 프레임 ECC 정합 중...")
        
        for i, frame in enumerate(self.baseline_frames):
            if i == 0:
                # 첫 번째 프레임은 정합 없이 그대로 저장
                cropped = frame[y:y+h, x:x+w]
                output_path = os.path.join(baseline_dir, f"baseline_{i:04d}.tiff")
                cv2.imwrite(output_path, cropped)
                print(f"  [0/{len(self.baseline_frames)}] 기준 프레임 저장")
                continue
            
            # 각 프레임을 첫 번째 프레임과 ECC 정합
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_roi_gray = frame_gray[y:y+h, x:x+w]
            frame_vesselness = self.preprocess_vesselness(frame_roi_gray)
            
            # ORB 초기 정합
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            orb = cv2.ORB_create(1000)
            kp1, des1 = orb.detectAndCompute(self.baseline_vesselness, None)
            kp2, des2 = orb.detectAndCompute(frame_vesselness, None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    M_init, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M_init is not None:
                        warp_matrix = M_init.astype(np.float32)
            
            # ECC 정밀 정합
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    self.baseline_vesselness,
                    frame_vesselness,
                    warp_matrix,
                    cv2.MOTION_EUCLIDEAN,
                    self.criteria
                )
            except cv2.error:
                pass  # 실패 시 ORB 결과 사용
            
            # 전체 이미지 워핑 후 크롭
            h_full, w_full = frame.shape[:2]
            warped = cv2.warpAffine(
                frame,
                warp_matrix,
                (w_full, h_full),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            aligned_roi = warped[y:y+h, x:x+w]
            
            output_path = os.path.join(baseline_dir, f"baseline_{i:04d}.tiff")
            cv2.imwrite(output_path, aligned_roi)
            
            if (i + 1) % 5 == 0 or i == len(self.baseline_frames) - 1:
                print(f"  진행: {i+1}/{len(self.baseline_frames)} 프레임 완료")
        
        print(f"✅ Baseline 저장 완료: {baseline_dir}")

    def align_window_frames_with_ecc(self, target_frame_dir, save_root_dir, max_frames=10):
        """Target 전체 프레임을 개별 ECC 정합 후 저장"""
        if self.roi_offset is None:
            print("❌ 먼저 set_baseline_from_frames를 호출하여 ROI를 설정해야 합니다.")
            return
        
        frames = load_frames_from_dir(target_frame_dir, max_frames=max_frames)
        if len(frames) == 0:
            print("❌ Target 프레임이 없습니다.")
            return

        x, y = self.roi_offset
        w, h = self.baseline_roi_size

        # Target 폴더 번호 추출
        target_name = os.path.basename(target_frame_dir)
        parts = target_name.split('_')
        if len(parts) >= 3:
            target_num = parts[2].lstrip('0') or '0'
        else:
            target_num = "unknown"
        
        aligned_dir = os.path.join(save_root_dir, f"Aligned_window_0to5s_{target_num}")
        os.makedirs(aligned_dir, exist_ok=True)

        print(f"\n📁 Target {len(frames)}개 프레임 ECC 정합 중...")
        
        for i, frame in enumerate(frames):
            # 각 프레임을 Baseline 첫 프레임과 ECC 정합
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_roi_gray = frame_gray[y:y+h, x:x+w]
            frame_vesselness = self.preprocess_vesselness(frame_roi_gray)
            
            # ORB 초기 정합
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            orb = cv2.ORB_create(1000)
            kp1, des1 = orb.detectAndCompute(self.baseline_vesselness, None)
            kp2, des2 = orb.detectAndCompute(frame_vesselness, None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    M_init, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M_init is not None:
                        warp_matrix = M_init.astype(np.float32)
            
            # ECC 정밀 정합
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    self.baseline_vesselness,
                    frame_vesselness,
                    warp_matrix,
                    cv2.MOTION_EUCLIDEAN,
                    self.criteria
                )
            except cv2.error:
                pass  # 실패 시 ORB 결과 사용
            
            # 전체 이미지 워핑 후 크롭
            h_full, w_full = frame.shape[:2]
            warped = cv2.warpAffine(
                frame,
                warp_matrix,
                (w_full, h_full),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            aligned_roi = warped[y:y+h, x:x+w]
            
            output_path = os.path.join(aligned_dir, f"aligned_{i:04d}.tiff")
            cv2.imwrite(output_path, aligned_roi)
            
            if (i + 1) % 5 == 0 or i == len(frames) - 1:
                print(f"  진행: {i+1}/{len(frames)} 프레임 완료")

        print(f"✅ Aligned 저장 완료: {aligned_dir}")


# --- 실행부 ---
if __name__ == "__main__":
    BASELINE_FRAMES = r"D:\VIDEO\M-24\Video_00056_00001_0to5s_tiff"
    TARGET_FRAMES   = r"D:\VIDEO\M-24\Video_00056_00052_2m30sto2m35s_tiff"
    # TARGET_FRAMES   = r"D:\VIDEO\M-24\Video_00056_00030_0to5s_tiff"
    # TARGET_FRAMES   = r"D:\VIDEO\M-24\Video_00056_00060_0to5s_tiff"

    # 루트 저장 폴더
    SAVE_ROOT_DIR = r"D:\VIDEO\M-24\ROI_window_ext_1_52_all"
    # SAVE_ROOT_DIR = r"D:\VIDEO\M-24\ROI_window_ext_1_30_all"
    # SAVE_ROOT_DIR = r"D:\VIDEO\M-24\ROI_window_0to5s_1_60_all"

    # 처리할 프레임 수 (테스트용으로 20개)
    MAX_FRAMES = 20

    aligner = UltraPreciseVesselAligner()

    print("=" * 60)
    print("🚀 혈관 이미지 정렬 시작 (모든 프레임 개별 ECC)")
    print(f"⚙️  처리 프레임 수: {MAX_FRAMES}장")
    print("=" * 60)

    # 1️⃣ Baseline 설정 (첫 번째 프레임에서 ROI 선택)
    print("\n[STEP 1] Baseline 프레임 로드 및 ROI 선택")
    aligner.set_baseline_from_frames(BASELINE_FRAMES, max_frames=MAX_FRAMES)

    # 2️⃣ Baseline 전체 프레임 개별 ECC 정합 후 저장
    print("\n[STEP 2] Baseline 프레임 개별 ECC 정합 및 저장")
    aligner.process_baseline_frames_with_ecc(SAVE_ROOT_DIR)

    # 3️⃣ Target 전체 프레임 개별 ECC 정합 후 저장
    print("\n[STEP 3] Target 프레임 개별 ECC 정합 및 저장")
    aligner.align_window_frames_with_ecc(TARGET_FRAMES, SAVE_ROOT_DIR, max_frames=MAX_FRAMES)
    
    print("\n" + "=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)