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

        self.fov_width_mm = fov_width_mm
        self.img_width_px = img_width_px
        self.um_per_px = (fov_width_mm * 1000) / img_width_px

        self.baseline_vesselness = None
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def visualize_orb_matches(self, base_img_path, target_img_path, save_path=None):
        """특징점 매칭 시각화 및 이미지 저장"""
        img1 = cv2.imread(base_img_path)
        img2 = cv2.imread(target_img_path)
        
        if img1 is None or img2 is None:
            print("❌ 시각화 실패: 이미지를 불러올 수 없습니다.")
            return

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=self.orb_nfeatures)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print("❌ 특징점을 추출할 수 없습니다.")
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 상위 50개 매칭 시각화
        match_viz = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 이미지 저장
        if save_path:
            cv2.imwrite(save_path, match_viz)
            print(f"📸 특징점 매칭 이미지가 저장되었습니다: {save_path}")

        window_name = "ORB Feature Matching Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, match_viz)
        print("💡 창을 닫으려면 아무 키나 누르세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_frames_from_video(self, video_path, start_sec=0, end_sec=5, max_frames=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        frames = []
        frame_numbers = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, min(end_frame, start_frame + max_frames)):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            frame_numbers.append(frame_idx)

        cap.release()
        return frames, frame_numbers

    def get_refined_mask(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        blur = cv2.GaussianBlur(a_channel, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
        kernel = np.ones((25, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
        main_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_cnt)
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask

    def calculate_masked_clarity(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = self.get_refined_mask(img_bgr)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        mag = cv2.magnitude(dx, dy)
        vessel_area_mag = mag[mask > 0]
        if len(vessel_area_mag) == 0: return 0
        top_k = max(int(len(vessel_area_mag) * 0.1), 1)
        return np.mean(np.sort(vessel_area_mag)[-top_k:])

    def preprocess_vesselness(self, img_gray):
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        return cv2.GaussianBlur(v_map, (3, 3), 0)

    def split_into_quadrants(self, roi_img):
        h, w = roi_img.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        return {
            1: roi_img[0:mid_h, 0:mid_w],
            2: roi_img[0:mid_h, mid_w:w],
            3: roi_img[mid_h:h, 0:mid_w],
            4: roi_img[mid_h:h, mid_w:w]
        }

    def find_first_clear_frame(self, frames):
        for i, frame in enumerate(frames):
            score = self.calculate_masked_clarity(frame)
            if score >= self.abs_threshold:
                print(f"✅ 첫 통과 프레임: {i}번 (선명도: {score:.2f})")
                return frame, i
        raise ValueError("선명도 통과한 프레임이 없습니다!")

    def set_baseline_roi(self, video_path, max_frames=10):
        frames, frame_nums = self.extract_frames_from_video(video_path, max_frames=max_frames)
        if not frames: raise ValueError("Baseline 프레임 추출 실패")
        first_clear_frame, idx = self.find_first_clear_frame(frames)

        window_name = "Select ROI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(window_name, first_clear_frame, False)
        cv2.destroyWindow(window_name)

        x, y, w, h = roi
        if w == 0 or h == 0: raise ValueError("ROI 미선택")

        self.roi_offset = (x, y)
        self.roi_size = (w, h)
        roi_gray = cv2.cvtColor(first_clear_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.baseline_vesselness = self.preprocess_vesselness(roi_gray)
        return frames, frame_nums, idx

    def get_roi_real_size(self):
        if self.roi_size is None: return 0, 0
        return self.roi_size[0] * self.um_per_px, self.roi_size[1] * self.um_per_px

    def get_quadrant_real_size(self):
        if self.roi_size is None: return 0, 0
        return (self.roi_size[0] // 2) * self.um_per_px, (self.roi_size[1] // 2) * self.um_per_px

    def compute_orb_init_warp(self, template_u8, image_u8):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        orb = cv2.ORB_create(self.orb_nfeatures)
        kp1, des1 = orb.detectAndCompute(template_u8, None)
        kp2, des2 = orb.detectAndCompute(image_u8, None)
        if des1 is None or des2 is None: return warp_matrix
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches or len(matches) < self.min_orb_matches: return warp_matrix
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M_init, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return M_init.astype(np.float32) if M_init is not None else warp_matrix

    def ecc_align_roi_by_warping_full_frame(self, frame_bgr):
        x, y = self.roi_offset
        w, h = self.roi_size
        roi_gray = cv2.cvtColor(frame_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        frame_v = self.preprocess_vesselness(roi_gray)
        warp_matrix = self.compute_orb_init_warp(self.baseline_vesselness, frame_v)
        cc = None
        try:
            cc, warp_matrix = cv2.findTransformECC(self.baseline_vesselness, frame_v, warp_matrix, cv2.MOTION_EUCLIDEAN, self.criteria)
        except: pass

        h_f, w_f = frame_bgr.shape[:2]
        warped_full = cv2.warpAffine(frame_bgr, warp_matrix, (w_f, h_f), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP)
        warped_mask = cv2.warpAffine(np.ones((h_f, w_f), dtype=np.uint8)*255, warp_matrix, (w_f, h_f), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        
        return warped_full[y:y+h, x:x+w], warped_mask[y:y+h, x:x+w], warp_matrix, cc

    def process_baseline(self, frames, frame_nums, first_idx, save_dir):
        x, y = self.roi_offset
        w, h = self.roi_size
        frame = frames[first_idx]
        roi_bgr = frame[y:y+h, x:x+w]
        
        quadrants = self.split_into_quadrants(roi_bgr)
        for q_num, q_img in quadrants.items():
            q_dir = os.path.join(save_dir, f"Quadrant_{q_num}")
            os.makedirs(q_dir, exist_ok=True)
            cv2.imwrite(os.path.join(q_dir, f"base_REF_q{q_num}.tiff"), q_img)

        return {
            "total_frames": 1, "whole_clarity_passed": 1, "whole_clarity_failed": 0, "ecc_failed": 0,
            "quadrant_success": {1: 1, 2: 1, 3: 1, 4: 1}, "quadrant_failed": {1: 0, 2: 0, 3: 0, 4: 0}
        }

    def process_target(self, video_path, save_dir, max_frames=150):
        frames, frame_nums = self.extract_frames_from_video(video_path, max_frames=max_frames)
        x, y = self.roi_offset
        w, h = self.roi_size
        
        stats = {
            "total_frames": 0, "whole_clarity_passed": 0, "whole_clarity_failed": 0, "ecc_failed": 0,
            "quadrant_success": {1: 0, 2: 0, 3: 0, 4: 0}, "quadrant_failed": {1: 0, 2: 0, 3: 0, 4: 0}
        }

        print(f"\n🚀 [Target] 첫 번째 매칭 프레임 탐색 중...")
        for i, (frame, f_num) in enumerate(zip(frames, frame_nums)):
            stats["total_frames"] += 1
            roi_org = frame[y:y+h, x:x+w]
            if self.calculate_masked_clarity(roi_org) < self.abs_threshold:
                stats["whole_clarity_failed"] += 1
                continue
            
            stats["whole_clarity_passed"] += 1
            try:
                aligned_roi, aligned_mask, _, cc = self.ecc_align_roi_by_warping_full_frame(frame)
                q_masks = self.split_into_quadrants(aligned_mask)
                
                if all([np.all(m == 255) for m in q_masks.values()]):
                    q_imgs = self.split_into_quadrants(aligned_roi)
                    for q_num, q_img in q_imgs.items():
                        q_dir = os.path.join(save_dir, f"Quadrant_{q_num}")
                        os.makedirs(q_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(q_dir, f"target_MATCH_q{q_num}.tiff"), q_img)
                        stats["quadrant_success"][q_num] = 1
                    print(f"🎯 매칭 성공: Frame {i} (CC: {cc if cc else 'ORB'})")
                    return stats
            except:
                stats["ecc_failed"] += 1
        return stats


if __name__ == "__main__":
    BASELINE_VIDEO = r"D:\VIDEO\M-24\Video_00056_00001.mp4"
    TARGET_VIDEO = r"D:\VIDEO\M-24\Video_00056_00030.mp4"
    BASE_RESULT_ROOT = r"D:\VIDEO\M-24"

    MAX_FRAMES = 150
    CLARITY_THRESHOLD = 10.0
    FOV_WIDTH_MM = 6.5
    IMG_WIDTH_PX = 3840

    start_time = time.time()
    aligner = QuadrantVesselAligner(abs_threshold=CLARITY_THRESHOLD)

    # 1. Baseline 설정
    baseline_frames, baseline_nums, first_idx = aligner.set_baseline_roi(BASELINE_VIDEO, max_frames=MAX_FRAMES)
    roi_w_um, roi_h_um = aligner.get_roi_real_size()
    quad_w_um, quad_h_um = aligner.get_quadrant_real_size()

    RESULT_ROOT = os.path.join(BASE_RESULT_ROOT, f"result_{roi_w_um:.2f}um_{roi_h_um:.2f}um")
    base_dir = os.path.join(RESULT_ROOT, f"Baseline_{quad_w_um:.2f}um")
    target_dir = os.path.join(RESULT_ROOT, f"Aligned_{quad_w_um:.2f}um")

    # 2. 처리
    b_stats = aligner.process_baseline(baseline_frames, baseline_nums, first_idx, base_dir)
    t_stats = aligner.process_target(TARGET_VIDEO, target_dir, max_frames=MAX_FRAMES)

    # 3. 리포트
    print("\n" + "="*50)
    print(f"🏁 처리 완료 (총 {time.time()-start_time:.2f}초)")
    print(f"📁 저장 폴더: {RESULT_ROOT}")
    print("="*50)

    # 4. 시각화 및 이미지 저장 (Q1 기준)
    s_base = os.path.join(base_dir, "Quadrant_1", "base_REF_q1.tiff")
    s_target = os.path.join(target_dir, "Quadrant_1", "target_MATCH_q1.tiff")
    s_viz = os.path.join(RESULT_ROOT, "ORB_matching_result.jpg") # 저장 경로

    if os.path.exists(s_base) and os.path.exists(s_target):
        aligner.visualize_orb_matches(s_base, s_target, save_path=s_viz)