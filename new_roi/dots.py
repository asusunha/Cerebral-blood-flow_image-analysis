import time
import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure


class QuadrantVesselAligner:
    def __init__(self, abs_threshold=20.0, orb_nfeatures=1000, min_orb_matches=10, fov_width_mm=6.5, img_width_px=3840):
        self.abs_threshold = abs_threshold
        self.orb_nfeatures = orb_nfeatures
        self.min_orb_matches = min_orb_matches

        self.roi_offset = None  # (x, y)
        self.roi_size = None  # (w, h)

        # 실제 거리 변환 비율
        self.fov_width_mm = fov_width_mm
        self.img_width_px = img_width_px
        self.um_per_px = (fov_width_mm * 1000) / img_width_px  # 픽셀당 마이크로미터

        # Baseline 기준(첫 통과 프레임 ROI)의 vesselness
        self.baseline_vesselness = None

        # ECC 알고리즘 종료 조건
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    # ----------------------------
    # I/O: video -> frames
    # ----------------------------
    def extract_frames_from_video(self, video_path, start_sec=0, end_sec=5, max_frames=10):
        """비디오에서 특정 구간의 프레임 추출 (상위 N개)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 비디오 정보: {width}x{height}, {fps:.2f} FPS")

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        frames = []
        frame_numbers = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, min(end_frame, start_frame + max_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_numbers.append(frame_idx)

        cap.release()
        print(f"✅ {len(frames)}개 프레임 추출 완료 (프레임 {start_frame}~{start_frame + len(frames) - 1})")
        return frames, frame_numbers

    # ----------------------------
    # Mask / Clarity
    # ----------------------------
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

        top_k = int(len(vessel_area_mag) * 0.1)
        top_k = max(top_k, 1)
        score = np.mean(np.sort(vessel_area_mag)[-top_k:])
        return score

    # ----------------------------
    # Vesselness for ECC
    # ----------------------------
    def preprocess_vesselness(self, img_gray):
        """ECC 정합용 Vesselness 맵 생성"""
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        return cv2.GaussianBlur(v_map, (3, 3), 0)

    # ----------------------------
    # Quadrants
    # ----------------------------
    def split_into_quadrants(self, roi_img):
        """ROI를 4분할 (좌상, 우상, 좌하, 우하)"""
        h, w = roi_img.shape[:2]
        mid_h, mid_w = h // 2, w // 2

        quadrants = {
            1: roi_img[0:mid_h, 0:mid_w],           # 좌상
            2: roi_img[0:mid_h, mid_w:w],           # 우상
            3: roi_img[mid_h:h, 0:mid_w],           # 좌하
            4: roi_img[mid_h:h, mid_w:w]            # 우하
        }
        return quadrants

    # ----------------------------
    # Baseline selection
    # ----------------------------
    def find_first_clear_frame(self, frames):
        """선명도 통과한 첫 프레임 찾기"""
        for i, frame in enumerate(frames):
            score = self.calculate_masked_clarity(frame)
            if score >= self.abs_threshold:
                print(f"✅ 첫 통과 프레임: {i}번 (선명도: {score:.2f})")
                return frame, i
        raise ValueError("선명도 통과한 프레임이 없습니다!")

    # --- Before (드래그 방식) ---
    # roi = cv2.selectROI(window_name, first_clear_frame, False)
    # x, y, w, h = roi

    # --- After (3점 클릭 및 ROI 자동 계산) ---
    def set_baseline_roi(self, video_path, max_frames=10):
        frames, frame_nums = self.extract_frames_from_video(video_path, max_frames=max_frames)
        first_clear_frame, idx = self.find_first_clear_frame(frames)
        
        # [수정] 정합을 위해 기준 프레임의 그레이스케일 이미지 저장
        self.baseline_gray = cv2.cvtColor(first_clear_frame, cv2.COLOR_BGR2GRAY)
        
        pts = []
        window_name = "Click 3 Points (Press ESC to finish)"
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
                pts.append([x, y])
                cv2.circle(display_img, (x, y), 15, (0, 255, 0), -1) # 점을 좀 더 크게 표시
                cv2.putText(display_img, str(len(pts)), (x+20, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow(window_name, display_img)

        display_img = first_clear_frame.copy()
        
        # [수정] 화면 크기 조절 가능하게 설정 (이전 코드 방식)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) # 적절한 모니터 크기로 리사이즈
        
        cv2.imshow(window_name, display_img)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("💡 혈관의 분기점 등 특징적인 3곳을 클릭하세요.")
        while len(pts) < 3:
            if cv2.waitKey(1) & 0xFF == 27: # ESC 누르면 중단
                break
        cv2.destroyWindow(window_name)

        if len(pts) < 3:
            raise ValueError("3개의 점이 모두 선택되지 않았습니다.")

        # 3점을 포함하는 최소 ROI 계산
        pts_arr = np.array(pts)
        x, y, w, h = cv2.boundingRect(pts_arr)
        
        self.roi_offset = (x, y)
        self.roi_size = (w, h)
        self.base_points = pts_arr.astype(np.float32).reshape(-1, 1, 2)
        
        # [참고] 기존 vesselness 로직 호환을 위해 template 저장 (선명도 계산용)
        roi_gray = self.baseline_gray[y:y+h, x:x+w]
        self.baseline_vesselness = self.preprocess_vesselness(roi_gray)
        
        return frames, frame_nums, idx

    # ----------------------------
    # 실제 크기 계산 함수
    # ----------------------------
    def get_roi_real_size(self):
        """전체 ROI의 실제 크기(um) 반환"""
        if self.roi_size is None:
            return 0, 0
        w, h = self.roi_size
        real_w_um = w * self.um_per_px
        real_h_um = h * self.um_per_px
        return real_w_um, real_h_um

    def get_quadrant_real_size(self):
        """4분할 ROI의 실제 크기(um) 반환"""
        if self.roi_size is None:
            return 0, 0
        w, h = self.roi_size
        quad_w = w // 2
        quad_h = h // 2
        real_w_um = quad_w * self.um_per_px
        real_h_um = quad_h * self.um_per_px
        return real_w_um, real_h_um

    # ----------------------------
    # ECC core: ORB init + ECC refine + warp FULL frame then crop ROI
    # ----------------------------
    def compute_orb_init_warp(self, template_u8, image_u8):
        """
        ORB 특징점 매칭으로 초기 warp(2x3)를 추정.
        template_u8: baseline_vesselness
        image_u8   : frame_vesselness
        """
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        orb = cv2.ORB_create(self.orb_nfeatures)
        kp1, des1 = orb.detectAndCompute(template_u8, None)
        kp2, des2 = orb.detectAndCompute(image_u8, None)

        if des1 is None or des2 is None:
            return warp_matrix

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if matches is None or len(matches) < self.min_orb_matches:
            return warp_matrix

        # distance 기준으로 정렬해서 상위 일부만 써도 됨(너무 느슨한 매칭 완화)
        matches = sorted(matches, key=lambda m: m.distance)
        # 상위 200개까지만 사용(경험적으로 안정성/속도 균형)
        matches = matches[:min(len(matches), 200)]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M_init, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10
        )

        if M_init is None:
            return warp_matrix

        return M_init.astype(np.float32)

    # --- Before (ORB + ECC 방식) ---
    # warp_matrix = self.compute_orb_init_warp(...)
    # cc, warp_matrix = cv2.findTransformECC(...)

    # --- After (3점 추적 방식) ---
    def ecc_align_roi_by_warping_full_frame(self, frame_bgr):
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 3점 추적 수행
        lk_params = dict(winSize=(31, 31), maxLevel=3, 
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.baseline_gray, curr_gray, 
                                                       self.base_points, None, **lk_params)

        if new_pts is None or status is None or np.sum(status) < 3:
            return None, None, None, 0.0 # 추적 실패 시 CC=0

        # 2. Affine Matrix 생성
        M = cv2.getAffineTransform(new_pts.astype(np.float32).reshape(3, 2), 
                                   self.base_points.astype(np.float32).reshape(3, 2))

        # 3. 전체 워핑
        h_f, w_f = frame_bgr.shape[:2]
        warped_full = cv2.warpAffine(frame_bgr, M, (w_f, h_f), flags=cv2.INTER_LANCZOS4)
        valid_mask = cv2.warpAffine(np.ones((h_f, w_f), dtype=np.uint8)*255, M, (w_f, h_f), 
                                    flags=cv2.INTER_NEAREST)

        # 4. [중요] 실제 유사도(CC) 계산 로직
        # 워핑된 결과의 ROI와 Baseline ROI 간의 상관계수를 계산합니다.
        x, y = self.roi_offset
        w, h = self.roi_size
        aligned_roi = warped_full[y:y+h, x:x+w]
        aligned_mask = valid_mask[y:y+h, x:x+w]
        
        # Baseline의 그레이스케일 ROI와 비교
        base_roi_gray = self.baseline_gray[y:y+h, x:x+w]
        curr_roi_gray = cv2.cvtColor(aligned_roi, cv2.COLOR_BGR2GRAY)
        
        # OpenCV의 CC 계산 함수 사용
        res = cv2.matchTemplate(curr_roi_gray, base_roi_gray, cv2.TM_CCOEFF_NORMED)
        actual_cc = res[0][0] # -1 ~ 1 사이의 값

        return aligned_roi, aligned_mask, M, actual_cc

    # ----------------------------
    # Pipeline: baseline processing
    # ----------------------------
    def process_baseline(self, frames, frame_nums, first_idx, save_dir):
        """
        Baseline 프레임 처리: 
        1) 전체 ROI 선명도 검사 
        2) (ORB+ECC) 전체 프레임 워핑 및 유효 영역 마스크 생성
        3) 사분면별 경계 침범 여부 검사 (ROI 유효성)
        4) 사분면별 선명도 검사 및 저장
        """
        x, y = self.roi_offset
        w, h = self.roi_size

        stats = {
            "total_frames": 0,
            "whole_clarity_passed": 0,
            "whole_clarity_failed": 0,
            "ecc_failed": 0,
            "border_out_failed": {1: 0, 2: 0, 3: 0, 4: 0}, # 경계 벗어남 통계 추가
            "quadrant_success": {1: 0, 2: 0, 3: 0, 4: 0},
            "quadrant_failed": {1: 0, 2: 0, 3: 0, 4: 0}
        }

        for quad_num in range(1, 5):
            quad_dir = os.path.join(save_dir, f"Quadrant_{quad_num}")
            os.makedirs(quad_dir, exist_ok=True)

        print(f"\n🚀 [Baseline] 처리 시작 (유효 영역 검증 로직 포함)")

        for i, (frame, frame_num) in enumerate(zip(frames, frame_nums)):
            stats["total_frames"] += 1
            roi_bgr_original = frame[y:y+h, x:x+w]

            print(f"\n📍 Frame {i:02d} (원본 #{frame_num})")

            # 1) 전체 ROI 선명도 검사 (정합 전 원본 위치 기준)
            whole_clarity = self.calculate_masked_clarity(roi_bgr_original)
            print(f"   🔍 전체 ROI 선명도: {whole_clarity:.2f}")

            if whole_clarity < self.abs_threshold:
                stats["whole_clarity_failed"] += 1
                print(f"   ❌ 전체 선명도 부족으로 프레임 제외 ({whole_clarity:.2f} < {self.abs_threshold})")
                continue

            stats["whole_clarity_passed"] += 1

            # 2) 기준 프레임 처리 (첫 번째 통과 프레임은 정합 없이 바로 사분면 분할)
            if i == first_idx:
                print(f"   ⭐ 기준 프레임 (정합 생략)")
                quadrants = self.split_into_quadrants(roi_bgr_original)
                for quad_num, quad_img in quadrants.items():
                    quad_clarity = self.calculate_masked_clarity(quad_img)
                    if quad_clarity >= self.abs_threshold:
                        save_path = os.path.join(save_dir, f"Quadrant_{quad_num}", f"base_f{i:02d}_q{quad_num}.tiff")
                        cv2.imwrite(save_path, quad_img)
                        stats["quadrant_success"][quad_num] += 1
                        print(f"     ✅ Q{quad_num}: 저장 완료 (선명도: {quad_clarity:.2f})")
                    else:
                        stats["quadrant_failed"][quad_num] += 1
                        print(f"     ❌ Q{quad_num}: 선명도 부족")
                continue

            # 3) 정합 및 사분면별 유효성 검사 (타겟 프레임들)
            try:
                # 수정된 ecc_align 함수는 (aligned_roi, aligned_mask, warp_matrix, cc)를 반환해야 함
                aligned_roi, aligned_mask, warp_matrix, cc = self.ecc_align_roi_by_warping_full_frame(frame)
                
                if cc is None:
                    print(f"   ⚠️ ECC 미수렴: ORB 결과 사용")
                else:
                    print(f"   ✅ 정합 성공 (cc={cc:.6f})")

                # 사분면 분할 (이미지와 유효 마스크 둘 다 분할)
                quad_imgs = self.split_into_quadrants(aligned_roi)
                quad_masks = self.split_into_quadrants(aligned_mask)

                for quad_num in range(1, 5):
                    target_quad_img = quad_imgs[quad_num]
                    target_quad_mask = quad_masks[quad_num]

                    # [추가된 로직] ROI 유효성 검사: 마스크에 0(경계 밖)이 포함되어 있는지 확인
                    if np.any(target_quad_mask == 0):
                        stats["border_out_failed"][quad_num] += 1
                        print(f"     ❌ Q{quad_num}: 이미지 경계를 벗어남 (제외)")
                        continue

                    # 선명도 검사
                    quad_clarity = self.calculate_masked_clarity(target_quad_img)
                    if quad_clarity >= self.abs_threshold:
                        save_path = os.path.join(save_dir, f"Quadrant_{quad_num}", f"base_f{i:02d}_q{quad_num}.tiff")
                        cv2.imwrite(save_path, target_quad_img)
                        stats["quadrant_success"][quad_num] += 1
                        print(f"     ✅ Q{quad_num}: 저장 완료 (선명도: {quad_clarity:.2f})")
                    else:
                        stats["quadrant_failed"][quad_num] += 1
                        print(f"     ❌ Q{quad_num}: 선명도 부족 ({quad_clarity:.2f})")

            except Exception as e:
                stats["ecc_failed"] += 1
                print(f"   ❌ 정합 파이프라인 실패: {e}")

        return stats

    # ----------------------------
    # Pipeline: target processing
    # ----------------------------
    def process_target(self, video_path, save_dir, max_frames=10):
        """
        Target 비디오 처리: 
        1) 전체 선명도 검사 
        2) (ORB+ECC) 전체프레임 워핑 및 유효 영역 마스크 생성 
        3) 사분면별 경계 침범 여부 검사 (ROI 유효성) 
        4) 사분면별 선명도 검사 및 저장
        """
        if self.roi_offset is None or self.baseline_vesselness is None:
            raise ValueError("먼저 Baseline을 처리하여 기준 ROI와 Template을 설정해야 합니다.")

        frames, frame_nums = self.extract_frames_from_video(video_path, max_frames=max_frames)

        x, y = self.roi_offset
        w, h = self.roi_size

        stats = {
            "total_frames": 0,
            "whole_clarity_passed": 0,
            "whole_clarity_failed": 0,
            "ecc_failed": 0,
            "border_out_failed": {1: 0, 2: 0, 3: 0, 4: 0}, # 잘림 발생 통계
            "quadrant_success": {1: 0, 2: 0, 3: 0, 4: 0},
            "quadrant_failed": {1: 0, 2: 0, 3: 0, 4: 0}
        }

        for quad_num in range(1, 5):
            quad_dir = os.path.join(save_dir, f"Quadrant_{quad_num}")
            os.makedirs(quad_dir, exist_ok=True)

        print(f"\n🚀 [Target] 처리 시작 (유효 영역 검증 로직 포함)")

        for i, (frame, frame_num) in enumerate(zip(frames, frame_nums)):
            stats["total_frames"] += 1
            
            # 정합 전 원본 위치에서의 ROI 추출 (전체 선명도 검사용)
            roi_bgr_original = frame[y:y+h, x:x+w]

            print(f"\n📍 Frame {i:02d} (원본 #{frame_num})")

            # 1) 전체 ROI 선명도 검사 (사전 필터링)
            whole_clarity = self.calculate_masked_clarity(roi_bgr_original)
            print(f"   🔍 전체 ROI 선명도: {whole_clarity:.2f}")

            if whole_clarity < self.abs_threshold:
                stats["whole_clarity_failed"] += 1
                print(f"   ❌ 전체 선명도 부족으로 프레임 제외 ({whole_clarity:.2f} < {self.abs_threshold})")
                continue

            stats["whole_clarity_passed"] += 1

            # 2) ORB+ECC 정합 및 유효 영역 마스크 획득
            try:
                # ecc_align_roi_by_warping_full_frame은 (aligned_roi, aligned_mask, warp_matrix, cc) 반환
                aligned_roi, aligned_mask, warp_matrix, cc = self.ecc_align_roi_by_warping_full_frame(frame)
                
                if cc is None:
                    print(f"   ⚠️ ECC 미수렴: ORB 결과로 진행")
                else:
                    print(f"   ✅ 정합 성공 (cc={cc:.6f})")

                # 3) 사분면 분할 (이미지 및 유효성 마스크)
                quad_imgs = self.split_into_quadrants(aligned_roi)
                quad_masks = self.split_into_quadrants(aligned_mask)

                for quad_num in range(1, 5):
                    target_quad_img = quad_imgs[quad_num]
                    target_quad_mask = quad_masks[quad_num]

                    # [ROI 유효성 검사] 사분면 마스크 내 0(경계 밖)이 있는지 확인
                    if np.any(target_quad_mask == 0):
                        stats["border_out_failed"][quad_num] += 1
                        print(f"     ❌ Q{quad_num}: 이미지 경계를 벗어남 (제외)")
                        continue

                    # 4) 사분면별 선명도 최종 검사 및 저장
                    quad_clarity = self.calculate_masked_clarity(target_quad_img)
                    if quad_clarity >= self.abs_threshold:
                        save_path = os.path.join(save_dir, f"Quadrant_{quad_num}", f"target_f{i:02d}_q{quad_num}.tiff")
                        cv2.imwrite(save_path, target_quad_img)
                        stats["quadrant_success"][quad_num] += 1
                        print(f"     ✅ Q{quad_num}: 저장 완료 (선명도: {quad_clarity:.2f})")
                    else:
                        stats["quadrant_failed"][quad_num] += 1
                        print(f"     ❌ Q{quad_num}: 선명도 부족 ({quad_clarity:.2f})")

            except Exception as e:
                stats["ecc_failed"] += 1
                print(f"   ❌ 정합 파이프라인 실패: {e}")

        return stats


# --- 메인 실행부 ---
if __name__ == "__main__":
    BASELINE_VIDEO = r"D:\VIDEO\M-24\Video_00056_00001.mp4"
    TARGET_VIDEO = r"D:\VIDEO\M-24\Video_00056_00030.mp4"
    BASE_RESULT_ROOT = r"D:\VIDEO\M-24"  # result 폴더가 생성될 기본 경로

    MAX_FRAMES = 15
    CLARITY_THRESHOLD = 10.0
    FOV_WIDTH_MM = 6.5
    IMG_WIDTH_PX = 3840

    # 전체 처리 시작 시간
    start_time = time.time()    

    print("=" * 70)
    print("🚀 혈관 영상 4분할 ORB+ECC 정합 시스템 시작")
    print(f"   - 처리 프레임 수: 상위 {MAX_FRAMES}개")
    print(f"   - 선명도 임계값: {CLARITY_THRESHOLD}")
    print(f"   - 로직: 전체ROI 선명도 → ORB초기값+ECC정합 → 전체프레임워핑 → 사분면 선명도 → 저장")
    print("=" * 70)

    aligner = QuadrantVesselAligner(
        abs_threshold=CLARITY_THRESHOLD,
        orb_nfeatures=1000,
        min_orb_matches=10,
        fov_width_mm=FOV_WIDTH_MM,
        img_width_px=IMG_WIDTH_PX
    )

    # STEP 1: Baseline ROI 설정
    print("\n[STEP 1] Baseline ROI 설정")
    baseline_frames, baseline_nums, first_idx = aligner.set_baseline_roi(
        BASELINE_VIDEO, max_frames=MAX_FRAMES
    )

    # ROI 실제 크기 계산
    roi_w_um, roi_h_um = aligner.get_roi_real_size()
    quad_w_um, quad_h_um = aligner.get_quadrant_real_size()

    # 폴더명 생성 (소수점 2자리까지)
    result_folder_name = f"result_{roi_w_um:.2f}um_{roi_h_um:.2f}um"
    aligned_folder_name = f"Aligned_{quad_w_um:.2f}um_{quad_h_um:.2f}um"
    baseline_folder_name = f"Baseline_{quad_w_um:.2f}um_{quad_h_um:.2f}um"

    RESULT_ROOT = os.path.join(BASE_RESULT_ROOT, result_folder_name)

    print(f"\n📏 전체 ROI 실제 크기: {roi_w_um:.2f} um x {roi_h_um:.2f} um")
    print(f"📏 4분할 ROI 실제 크기: {quad_w_um:.2f} um x {quad_h_um:.2f} um")
    print(f"📁 결과 저장 폴더: {RESULT_ROOT}")

    # STEP 2: Baseline 처리
    print("\n[STEP 2] Baseline 처리")
    baseline_dir = os.path.join(RESULT_ROOT, baseline_folder_name)
    baseline_stats = aligner.process_baseline(
        baseline_frames, baseline_nums, first_idx, baseline_dir
    )

    # STEP 3: Target 처리
    print("\n[STEP 3] Target 처리")
    target_dir = os.path.join(RESULT_ROOT, aligned_folder_name)
    target_stats = aligner.process_target(
        TARGET_VIDEO, target_dir, max_frames=MAX_FRAMES
    )

    # 최종 리포트
    print("\n" + "=" * 70)
    print("🏁 최종 처리 결과")
    print("=" * 70)

    print(f"\n📊 Baseline 결과")
    print(f"   총 프레임: {baseline_stats['total_frames']}개")
    print(f"   전체 ROI 선명도 통과: {baseline_stats['whole_clarity_passed']}개")
    print(f"   전체 ROI 선명도 실패: {baseline_stats['whole_clarity_failed']}개")
    print(f"   정합 파이프라인 실패: {baseline_stats['ecc_failed']}개")
    print(f"\n   📁 사분면별 저장 결과:")
    for quad_num in range(1, 5):
        success = baseline_stats['quadrant_success'][quad_num]
        failed = baseline_stats['quadrant_failed'][quad_num]
        print(f"      Quadrant {quad_num}: ✅ {success}장 / ❌ {failed}장 제외")

    print(f"\n📊 Target 결과")
    print(f"   총 프레임: {target_stats['total_frames']}개")
    print(f"   전체 ROI 선명도 통과: {target_stats['whole_clarity_passed']}개")
    print(f"   전체 ROI 선명도 실패: {target_stats['whole_clarity_failed']}개")
    print(f"   정합 파이프라인 실패: {target_stats['ecc_failed']}개")
    print(f"\n   📁 사분면별 저장 결과:")
    for quad_num in range(1, 5):
        success = target_stats['quadrant_success'][quad_num]
        failed = target_stats['quadrant_failed'][quad_num]
        print(f"      Quadrant {quad_num}: ✅ {success}장 / ❌ {failed}장 제외")

    print(f"\n💾 저장 경로: {RESULT_ROOT}")
    print(f"   📏 전체 ROI 크기: {roi_w_um:.2f} x {roi_h_um:.2f} um")
    print(f"   📏 4분할 ROI 크기: {quad_w_um:.2f} x {quad_h_um:.2f} um")

    # 전체 종료 완료 시간
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"\n⏱️  총 처리 시간: {minutes}분 {seconds:.2f}초")

    print("=" * 70)