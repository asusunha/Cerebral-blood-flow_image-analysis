'''
# 입력
Baseline Image: Video_00056_00001_0s
Target Image: Video_00056_00030_0s

# 출력
Aligned_ROI 폴더
- crop된 Baseline Image
- crop된 Aligned Image

# 결과
정확도 매우 떨어짐
'''

import cv2
import numpy as np
import os

class UltraPreciseVesselAligner:
    def __init__(self, baseline_roi_size=(1648, 1424), final_crop_size=(1024, 1024)):
        """
        baseline_roi_size: baseline 이미지에서 추출할 ROI 크기 (width, height)
        final_crop_size: 최종 저장할 크롭 크기 (width, height)
        """
        self.baseline_roi_size = baseline_roi_size
        self.final_crop_size = final_crop_size
        self.baseline_gray = None
        self.baseline_roi_gray = None
        self.baseline_roi_color = None
        self.roi_offset = None
        # 정밀 정렬 종료 조건 (더 관대하게 조정)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    def set_baseline(self, img_path):
        """기준 프레임을 설정하고 중앙에서 ROI를 추출합니다."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"기준 이미지를 찾을 수 없습니다: {img_path}")
        
        # 전체 이미지를 그레이스케일로 변환
        self.baseline_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 중앙에서 ROI 추출
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        roi_w, roi_h = self.baseline_roi_size
        x1 = center_x - (roi_w // 2)
        y1 = center_y - (roi_h // 2)
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        # 이미지 범위를 벗어나지 않도록 보정
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # ROI 추출
        self.baseline_roi_gray = self.baseline_gray[y1:y2, x1:x2]
        self.baseline_roi_color = img[y1:y2, x1:x2]
        self.roi_offset = (x1, y1)
        
        print(f"✅ 기준 프레임 설정 완료: {os.path.basename(img_path)}")
        print(f"   - 원본 크기: {w}x{h}")
        print(f"   - ROI 크기: {self.baseline_roi_gray.shape[1]}x{self.baseline_roi_gray.shape[0]}")
        print(f"   - ROI 위치: ({x1}, {y1}) ~ ({x2}, {y2})")
        
        return img.shape, self.baseline_roi_gray.shape

    def show_baseline_roi(self, save_path=None):
        """추출된 baseline ROI를 화면에 표시합니다."""
        if self.baseline_roi_color is None:
            print("❌ 먼저 set_baseline()을 호출하여 기준 프레임을 설정하세요.")
            return None
        
        print("\n📸 Baseline ROI 표시 중...")
        print(f"   - ROI 크기: {self.baseline_roi_color.shape[1]}x{self.baseline_roi_color.shape[0]}")
        
        if save_path:
            cv2.imwrite(save_path, self.baseline_roi_color)
            print(f"   - 저장 완료: {save_path}")
        
        cv2.imshow("Baseline ROI (Press any key to continue)", self.baseline_roi_color)
        print("   - 창을 닫으려면 아무 키나 누르세요...\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.baseline_roi_color

    def preprocess_for_alignment(self, img_gray):
        """정렬 성능 향상을 위한 전처리"""
        # 1. 히스토그램 평활화로 대비 정규화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        
        # 2. 가우시안 블러로 노이즈 제거 (미세한 노이즈가 정렬 방해)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return denoised

    def align_and_crop_v1_direct(self, target_img_path, save_path):
        """
        방법 1: 직접 정렬 (ROI를 Target 전체와 비교)
        - 빠르지만 ROI가 Target보다 크면 실패 가능
        """
        if self.baseline_roi_gray is None:
            print("❌ 먼저 set_baseline()을 호출하세요.")
            return None
        
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            print(f"❌ 대상 이미지를 불러올 수 없습니다: {target_img_path}")
            return None
        
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        # 전처리
        baseline_processed = self.preprocess_for_alignment(self.baseline_roi_gray)
        target_processed = self.preprocess_for_alignment(target_gray)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        print(f"🔄 [방법 1] 직접 정렬 시도 중... (대상: {os.path.basename(target_img_path)})")
        try:
            (cc, warp_matrix) = cv2.findTransformECC(
                baseline_processed, 
                target_processed, 
                warp_matrix, 
                cv2.MOTION_EUCLIDEAN, 
                self.criteria, 
                None, 
                1
            )
            
            print(f"   ✅ 정렬 성공! 상관계수: {cc:.6f}")
            
            h, w = target_gray.shape
            aligned_img = cv2.warpAffine(
                target_img, 
                warp_matrix, 
                (w, h), 
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP
            )

            x1, y1 = self.roi_offset
            roi_w, roi_h = self.baseline_roi_size
            aligned_roi = aligned_img[y1:y1+roi_h, x1:x1+roi_w]
            
            cv2.imwrite(save_path, aligned_roi)
            print(f"   💾 저장 완료: {save_path}\n")
            
            return aligned_roi

        except cv2.error as e:
            print(f"   ❌ 방법 1 실패: {str(e)[:100]}\n")
            return None

    def align_and_crop_v2_roi_matching(self, target_img_path, save_path):
        """
        방법 2: ROI 영역 매칭 (Target에서 동일 위치 추출 후 정렬)
        - Target과 Baseline이 거의 정렬되어 있을 때 유리
        """
        if self.baseline_roi_gray is None:
            print("❌ 먼저 set_baseline()을 호출하세요.")
            return None
        
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            print(f"❌ 대상 이미지를 불러올 수 없습니다: {target_img_path}")
            return None
        
        # Target에서 동일 위치 ROI 먼저 추출
        x1, y1 = self.roi_offset
        roi_w, roi_h = self.baseline_roi_size
        x2, y2 = x1 + roi_w, y1 + roi_h
        
        target_h, target_w = target_img.shape[:2]
        if x2 > target_w or y2 > target_h:
            print(f"   ❌ Target 이미지가 너무 작습니다. ROI 범위 초과")
            print(f"      Target: {target_w}x{target_h}, 필요: {x2}x{y2}")
            return None
        
        target_roi_color = target_img[y1:y2, x1:x2]
        target_roi_gray = cv2.cvtColor(target_roi_color, cv2.COLOR_BGR2GRAY)
        
        # 전처리
        baseline_processed = self.preprocess_for_alignment(self.baseline_roi_gray)
        target_processed = self.preprocess_for_alignment(target_roi_gray)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        print(f"🔄 [방법 2] ROI 간 정렬 시도 중... (대상: {os.path.basename(target_img_path)})")
        try:
            (cc, warp_matrix) = cv2.findTransformECC(
                baseline_processed,
                target_processed,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                self.criteria,
                None,
                1
            )
            
            print(f"   ✅ 정렬 성공! 상관계수: {cc:.6f}")
            
            aligned_roi = cv2.warpAffine(
                target_roi_color,
                warp_matrix,
                (roi_w, roi_h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP
            )
            
            cv2.imwrite(save_path, aligned_roi)
            print(f"   💾 저장 완료: {save_path}\n")
            
            return aligned_roi

        except cv2.error as e:
            print(f"   ❌ 방법 2 실패: {str(e)[:100]}\n")
            return None

    def align_and_crop_v3_feature_based(self, target_img_path, save_path):
        """
        방법 3: 특징점 기반 정렬 (ORB + RANSAC)
        - ECC가 실패할 때 대안으로 사용
        - 큰 변형에도 강인함
        """
        if self.baseline_roi_gray is None:
            print("❌ 먼저 set_baseline()을 호출하세요.")
            return None
        
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            print(f"❌ 대상 이미지를 불러올 수 없습니다: {target_img_path}")
            return None
        
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        print(f"🔄 [방법 3] 특징점 기반 정렬 시도 중... (대상: {os.path.basename(target_img_path)})")
        
        # ORB 특징점 검출기
        orb = cv2.ORB_create(nfeatures=5000)
        
        kp1, des1 = orb.detectAndCompute(self.baseline_roi_gray, None)
        kp2, des2 = orb.detectAndCompute(target_gray, None)
        
        if des1 is None or des2 is None:
            print("   ❌ 특징점을 찾을 수 없습니다.")
            return None
        
        # BFMatcher로 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:
            print(f"   ❌ 매칭점이 부족합니다 ({len(matches)}개)")
            return None
        
        print(f"   - 매칭점 수: {len(matches)}개")
        
        # 상위 매칭점으로 변환 행렬 계산
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        
        # RANSAC으로 아웃라이어 제거
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, 
                                               ransacReprojThreshold=5.0)
        
        if M is None:
            print("   ❌ 변환 행렬을 계산할 수 없습니다.")
            return None
        
        print(f"   ✅ 정렬 성공! 인라이어: {mask.sum()}/{len(matches[:100])}개")
        
        h, w = target_gray.shape
        aligned_img = cv2.warpAffine(target_img, M, (w, h), flags=cv2.INTER_LANCZOS4)
        
        x1, y1 = self.roi_offset
        roi_w, roi_h = self.baseline_roi_size
        aligned_roi = aligned_img[y1:y1+roi_h, x1:x1+roi_w]
        
        cv2.imwrite(save_path, aligned_roi)
        print(f"   💾 저장 완료: {save_path}\n")
        
        return aligned_roi

    def align_and_crop(self, target_img_path, save_path):
        """
        스마트 정렬: 여러 방법을 순차적으로 시도
        """
        print("="*60)
        print("🎯 다중 전략 정렬 시작")
        print("="*60)
        
        # 방법 1 시도
        result = self.align_and_crop_v2_roi_matching(target_img_path, save_path)
        if result is not None:
            return result
        
        # 방법 2 시도
        result = self.align_and_crop_v1_direct(target_img_path, save_path)
        if result is not None:
            return result
        
        # 방법 3 시도 (최후의 수단)
        result = self.align_and_crop_v3_feature_based(target_img_path, save_path)
        if result is not None:
            return result
        
        print("❌ 모든 정렬 방법이 실패했습니다.")
        return None

    def visualize_baseline_roi(self, img_path, save_preview=True):
        """baseline 이미지에 ROI 영역을 표시하여 확인합니다."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 이미지를 불러올 수 없습니다: {img_path}")
            return
        
        if self.roi_offset is None:
            print("❌ 먼저 set_baseline()을 호출하세요.")
            return
        
        x1, y1 = self.roi_offset
        roi_w, roi_h = self.baseline_roi_size
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        preview = img.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.line(preview, (x1 + roi_w//2, y1), (x1 + roi_w//2, y2), (255, 0, 0), 2)
        cv2.line(preview, (x1, y1 + roi_h//2), (x2, y1 + roi_h//2), (255, 0, 0), 2)
        
        if save_preview:
            preview_path = img_path.replace('.tiff', '_ROI_Preview.jpg')
            cv2.imwrite(preview_path, preview)
            print(f"📸 ROI 미리보기 저장: {preview_path}")
        
        return preview


# --- 실행 설정 ---
if __name__ == "__main__":
    BASELINE_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00001'
    TARGET_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030'
    
    BASELINE_NAME = 'Video_00056_00001_0s.tiff'
    TARGET_NAME = 'Video_00056_00030_0s.tiff'
    
    SAVE_DIR = os.path.join(TARGET_DIR, "Aligned_ROI")
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)

    aligner = UltraPreciseVesselAligner(
        baseline_roi_size=(1648, 1424),
        final_crop_size=(1648, 1424)
    )
    
    baseline_full_path = os.path.join(BASELINE_DIR, BASELINE_NAME)
    aligner.set_baseline(baseline_full_path)
    
    baseline_roi_save_path = os.path.join(SAVE_DIR, "Baseline_ROI.tiff")
    aligner.show_baseline_roi(save_path=baseline_roi_save_path)
    
    preview = aligner.visualize_baseline_roi(baseline_full_path, save_preview=True)
    
    target_full_path = os.path.join(TARGET_DIR, TARGET_NAME)
    output_path = os.path.join(SAVE_DIR, "Aligned_" + TARGET_NAME)
    
    result = aligner.align_and_crop(target_full_path, output_path)

    if result is not None:
        comparison = np.hstack([aligner.baseline_roi_color, result])
        cv2.imshow("Left: Baseline ROI | Right: Aligned Target ROI", comparison)
        print("\n✅ 정렬 완료! 창을 닫으려면 아무 키나 누르세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n💡 해결 방안:")
        print("   1. ROI 크기를 줄여보세요 (예: 1200x1000)")
        print("   2. 두 이미지가 너무 다른지 육안으로 확인하세요")
        print("   3. Target 이미지 크기가 충분한지 확인하세요")