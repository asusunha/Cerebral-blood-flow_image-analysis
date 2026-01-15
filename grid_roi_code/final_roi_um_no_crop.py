'''
# [1단계] 기준점 설정 (Baseline Setup)
비디오에서 선명도 기준을 통과한 첫 번째 프레임을 찾습니다.

사용자가 분석하고 싶은 **관심 영역(ROI)**을 마우스로 드래그하여 선택합니다.

이 영역의 혈관 특징(Sato Filter 기반)을 추출해, 나중에 다른 프레임들을 정렬할 때 쓸 **'기준 지도(Template)'**를 만듭니다.

# [2단계] 선명도 기반 1차 필터링 (Pre-Filtering)
비디오의 각 프레임을 읽어와서 위에서 설명한 방식으로 선명도를 계산합니다.

통과: 정합 단계로 이동.

탈락: "선명도 부족" 메시지를 출력하고 해당 프레임은 버립니다.

# [3단계] 2단계 정밀 정합 (ORB + ECC Alignment)
1차(ORB): 특징점 매칭을 통해 큰 움직임(카메라 흔들림 등)을 빠르게 잡습니다.

2차(ECC): 마이크로미터 단위의 미세한 오차를 줄이기 위해 픽셀 단위로 정밀하게 위치를 보정합니다. (MOTION_EUCLIDEAN 모델 사용)

# [4단계] 유효성 검사 및 최종 저장 (Post-Filtering & Save)
경계 검사 (Border Check): 정합을 위해 이미지를 이동시켰을 때, 우리가 선택한 ROI 영역이 화면 밖으로 잘려 나가는지 확인합니다. (잘린 부분이 1픽셀이라도 있으면 데이터 오염 방지를 위해 제외)

저장: 모든 관문을 통과한 '깨끗하고 정렬된' 이미지들만 tiff 형식으로 저장됩니다.
'''
import time
import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure


class QuadrantVesselAligner:
    def __init__(self, abs_threshold=20.0, cc_threshold=0.35, orb_nfeatures=1000, min_orb_matches=10, fov_width_mm=6.5, img_width_px=3840):
        self.abs_threshold = abs_threshold
        self.cc_threshold = cc_threshold
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

    def set_baseline_roi(self, video_path, max_frames=10):
        """Baseline 비디오에서 첫 통과 프레임으로 ROI 설정"""
        frames, frame_nums = self.extract_frames_from_video(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError("Baseline 프레임을 추출할 수 없습니다.")

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

        # Baseline ROI vesselness 생성 (정합의 template)
        roi_bgr = first_clear_frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(roi_gray)

        print(f"✅ Baseline ROI 설정 완료: x={x}, y={y}, w={w}, h={h}")
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

    def ecc_align_roi_by_warping_full_frame(self, frame_bgr):
        """
        1) ROI crop → vesselness
        2) ORB 초기값 추정
        3) ECC로 정밀 정합(MOTION_EUCLIDEAN)
        4) 전체 프레임 warpAffine(+WARP_INVERSE_MAP)
        5) ROI를 다시 crop하여 aligned_roi 반환

        return: (aligned_roi_bgr, warp_matrix, cc)
        """
        if self.roi_offset is None or self.roi_size is None or self.baseline_vesselness is None:
            raise ValueError("Baseline ROI/template가 설정되지 않았습니다.")

        x, y = self.roi_offset
        w, h = self.roi_size

        roi_bgr = frame_bgr[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        frame_v = self.preprocess_vesselness(roi_gray)

        # ORB로 초기 warp 추정
        warp_matrix = self.compute_orb_init_warp(self.baseline_vesselness, frame_v)

        # ECC로 미세 정합
        cc = None
        try:
            cc, warp_matrix = cv2.findTransformECC(
                self.baseline_vesselness,
                frame_v,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                self.criteria
            )
        except cv2.error:
            # ECC 실패 시: ORB 초기값(또는 I) 그대로 사용
            cc = None

        # 전체 프레임 워핑 후 ROI 크롭 (설명했던 방식)
        h_full, w_full = frame_bgr.shape[:2]
    
        # 1. 실제 영상 워핑
        warped_full = cv2.warpAffine(
            frame_bgr, warp_matrix, (w_full, h_full),
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0) # 잘린 곳은 0으로
        )
        
        # 2. 유효 영역 검증을 위한 가상 마스크 생성 (Full Frame 크기)
        # 이미지의 유효한 영역만 255로 채워진 마스크
        valid_mask_full = np.ones((h_full, w_full), dtype=np.uint8) * 255
        warped_mask = cv2.warpAffine(
            valid_mask_full, warp_matrix, (w_full, h_full),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0 # 벗어난 곳은 0
        )

        aligned_roi = warped_full[y:y+h, x:x+w]
        aligned_mask = warped_mask[y:y+h, x:x+w] # ROI 영역에 해당하는 마스크 추출
        
        return aligned_roi, aligned_mask, warp_matrix, cc

    # ----------------------------
    # Pipeline: baseline processing (수정본)
    # ----------------------------
    def process_baseline(self, frames, frame_nums, first_idx, save_dir):
        """
        Baseline 프레임 처리: 사분면 분할 없이 전체 ROI 하나로 저장
        """
        x, y = self.roi_offset
        w, h = self.roi_size
        os.makedirs(save_dir, exist_ok=True)

        stats = {
            "total_frames": 0,
            "whole_clarity_passed": 0,
            "whole_clarity_failed": 0,
            "ecc_failed": 0,
            "border_out_failed": 0,
            "success_count": 0
        }

        print(f"\n🚀 [Baseline] 처리 시작 (단일 ROI 통합 저장)")

        for i, (frame, frame_num) in enumerate(zip(frames, frame_nums)):
            stats["total_frames"] += 1
            roi_bgr_original = frame[y:y+h, x:x+w]

            print(f"\n📍 Frame {i:02d} (원본 #{frame_num})")

            # 1) 전체 ROI 선명도 검사
            whole_clarity = self.calculate_masked_clarity(roi_bgr_original)
            if whole_clarity < self.abs_threshold:
                stats["whole_clarity_failed"] += 1
                print(f"   ❌ 선명도 부족 ({whole_clarity:.2f} < {self.abs_threshold})")
                continue

            stats["whole_clarity_passed"] += 1

            # 2) 기준 프레임 (정합 생략)
            if i == first_idx:
                save_path = os.path.join(save_dir, f"base_f{i:02d}_original.tiff")
                cv2.imwrite(save_path, roi_bgr_original)
                stats["success_count"] += 1
                print(f"   ⭐ 기준 프레임 저장 완료")
                continue

            # 3) 정합 및 유효성 검사
            try:
                aligned_roi, aligned_mask, _, cc = self.ecc_align_roi_by_warping_full_frame(frame)
                
                if cc is None or cc < self.cc_threshold:
                    stats["ecc_failed"] += 1
                    print(f"   ❌ ECC 정합 실패 (cc={cc})")
                    continue

                # ROI 전체 유효성 검증 (검은색 영역 포함 여부)
                if np.any(aligned_mask == 0):
                    stats["border_out_failed"] += 1
                    print(f"   ❌ 이미지 경계 벗어남 (제외)")
                    continue

                # 최종 저장
                save_path = os.path.join(save_dir, f"base_f{i:02d}_aligned.tiff")
                cv2.imwrite(save_path, aligned_roi)
                stats["success_count"] += 1
                print(f"   ✅ 정합 및 저장 완료 (cc={cc:.4f})")

            except Exception as e:
                stats["ecc_failed"] += 1
                print(f"   ❌ 에러 발생: {e}")

        return stats

    # ----------------------------
    # Pipeline: target processing (수정본)
    # ----------------------------
    def process_target(self, video_path, save_dir, max_frames=10):
        """
        Target 비디오 처리: 사분면 분할 없이 전체 ROI 하나로 저장
        """
        frames, frame_nums = self.extract_frames_from_video(video_path, max_frames=max_frames)
        x, y = self.roi_offset
        w, h = self.roi_size
        os.makedirs(save_dir, exist_ok=True)

        stats = {
            "total_frames": 0,
            "whole_clarity_passed": 0,
            "whole_clarity_failed": 0,
            "ecc_failed": 0,
            "border_out_failed": 0,
            "success_count": 0
        }

        print(f"\n🚀 [Target] 처리 시작 (단일 ROI 통합 저장)")

        for i, (frame, frame_num) in enumerate(zip(frames, frame_nums)):
            stats["total_frames"] += 1
            roi_bgr_original = frame[y:y+h, x:x+w]

            print(f"\n📍 Frame {i:02d} (원본 #{frame_num})")

            # 1) 전체 ROI 선명도 검사
            whole_clarity = self.calculate_masked_clarity(roi_bgr_original)
            if whole_clarity < self.abs_threshold:
                stats["whole_clarity_failed"] += 1
                print(f"   ❌ 선명도 부족 ({whole_clarity:.2f})")
                continue

            stats["whole_clarity_passed"] += 1

            # 2) 정합
            try:
                aligned_roi, aligned_mask, _, cc = self.ecc_align_roi_by_warping_full_frame(frame)
                
                if cc is None or cc < self.cc_threshold:
                    stats["ecc_failed"] += 1
                    print(f"   ❌ ECC 실패 (cc={cc})")
                    continue

                # 3) 경계 검사
                if np.any(aligned_mask == 0):
                    stats["border_out_failed"] += 1
                    print(f"   ❌ 경계 벗어남")
                    continue

                # 4) 저장
                save_path = os.path.join(save_dir, f"target_f{i:02d}_aligned.tiff")
                cv2.imwrite(save_path, aligned_roi)
                stats["success_count"] += 1
                print(f"   ✅ 저장 완료 (cc={cc:.4f})")

            except Exception as e:
                stats["ecc_failed"] += 1
                print(f"   ❌ 에러 발생: {e}")

        return stats


# --- 메인 실행부 ---
if __name__ == "__main__":
    BASELINE_VIDEO = r"D:\VIDEO\M-26\Video_00058_00044.mp4"
    TARGET_VIDEO = r"D:\VIDEO\M-26\Video_00058_00093.mp4"
    BASE_RESULT_ROOT = r"D:\VIDEO\M-26\final_roi_um2_no_crop"  # result 폴더가 생성될 기본 경로

    MAX_FRAMES = 15
    CLARITY_THRESHOLD = 10.0
    FOV_WIDTH_MM = 6.5
    IMG_WIDTH_PX = 3840
    CC_THRESHOLD = 0.35

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
        cc_threshold = CC_THRESHOLD,
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

    roi_w_um, roi_h_um = aligner.get_roi_real_size()

    # 폴더명 생성 (Quadrant 관련 문구 제거)
    result_folder_name = f"result_{roi_w_um:.2f}um_{roi_h_um:.2f}um_44_93"
    RESULT_ROOT = os.path.join(BASE_RESULT_ROOT, result_folder_name)

    # STEP 2: Baseline 처리
    print("\n[STEP 2] Baseline 처리")
    baseline_dir = os.path.join(RESULT_ROOT, "Baseline_ROI")
    baseline_stats = aligner.process_baseline(baseline_frames, baseline_nums, first_idx, baseline_dir)

    # STEP 3: Target 처리
    print("\n[STEP 3] Target 처리")
    target_dir = os.path.join(RESULT_ROOT, "Aligned_ROI")
    target_stats = aligner.process_target(TARGET_VIDEO, target_dir, max_frames=MAX_FRAMES)

    # 최종 리포트 출력
    print("\n" + "=" * 70)
    print("🏁 최종 처리 결과")
    print(f"📊 Baseline: 성공 {baseline_stats['success_count']} / 실패(경계) {baseline_stats['border_out_failed']}")
    print(f"📊 Target  : 성공 {target_stats['success_count']} / 실패(경계) {target_stats['border_out_failed']}")
    print(f"💾 경로: {RESULT_ROOT}")
    print("=" * 70)