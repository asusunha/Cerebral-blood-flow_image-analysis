import cv2 # 영상처리 핵심 라이브러리
import numpy as np # 배열 연산
import matplotlib.pyplot as plt # 시각화
import os # 경로 처리를 위해 추가
# 혈관의 골격 추출로 중심선 파악
from skimage.filters import sato # Sato 필터 (곡선 구조 검출, 현재 미사용)
from skimage.morphology import skeletonize # 중심선 추출
# 거리 변환으로 혈관 두께 계산
from scipy.ndimage import distance_transform_edt # 거리 변환 (직경 측정용)

class FinalVesselSystem:
    # 격자 간격 설정 (현재 미사용)
    def __init__(self, grid_spacing=80):
        self.grid_spacing = grid_spacing

    # 색상 기반 ROI 추출
    def get_color_roi(self, raw_img):
        """[로직 1] 색상 기반 ROI: 붉은색 채널을 이용하여 노이즈 영역 제거"""
        # Lab 색공간에서 'a' 채널(빨강-초록)은 혈관 영역 탐지에 매우 강력함
        # a 채널: 망막의 붉은 혈관 조직은 높은 값, 검은 배경은 중간값(128)
        # RGB vs Lab: 조명 변화에 강하고, 색상 분력이 명확
        lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        
        # 1. 붉은 성분 추출을 위한 임계값 처리
        # 21x21 커널: 큰 커널로 노이즈 제거하면서 부드러운 경계 생성
        # 135임계값: 붉은 조직은 통과, 배경은 차단
        # 결과: 망막 영역은 흰색(225), 배경은 검은색(0)
        blur = cv2.GaussianBlur(a_channel, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY) # 135는 붉은색 감도 조절용
        
        # 2. 강력한 모폴로지로 조각난 영역 합치기
        # CLOSE 연산: 팽창 -> 침식 순서로 진행
        # 효과: 작은 구멍 메우기, 끊어진 영역 연결
        # 25x25 커널: 망막 내부의 혈관 음영으로 생긴 작은 틈새 메우기
        kernel = np.ones((25, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 3. 가장 큰 덩어리만 선택 후 매끄럽게 연결
        # findContours: 흰색 영역의 외곽선 탐지
        # 혈관 음영으로 생긴 오목한 부분을 매끄럽게 연결
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        
        main_cnt = max(contours, key=cv2.contourArea) # 면적이 가장 큰 윤곽선
        hull = cv2.convexHull(main_cnt) # 볼록 껍질
        
        # 4. 최종 마스크 생성
        # 검은 캔버스에 hull 영역을 흰색으로 채운 이진 마스크 반환
        mask = np.zeros(raw_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask, hull

    # Vessel Map: CLAHE+적응형 이진화 -> 혈관 위치와 두께
    # 입력: 그레이스케일
    # 출력: 이진이미지
    def get_clean_vessels(self, raw_img, roi_mask, block_size=21, c_value=7):
        """[로직 2] 혈관 정밀 추출: CLAHE 대비 강화 + 적응형 이진화"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # 미세혈관 대비 강화 (CLAHE 대비 강화)
        '''
        이미지를 8x8 타일로 분할 -> 각 타일마다 독립적으로 히스토그램 평탄화 수행
        (일반 히스토그램 평탄화는 밝은 영역만 강조하는 반면, CLAHE는 어두운 영역의 혈관도 선명하게 부각)
        '''
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        
        # 혈관 추출 (적응형 이진화)
        vessel = cv2.adaptiveThreshold(enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 가우시간 가중 평균
                cv2.THRESH_BINARY_INV, # 어두운 픽셀을 흰색으로 반전
                block_size, # 클수록 큰 구조 검출 / 작을수록 미세 구조 검출
                c_value) # 클수록 더 많은 혈관 검출 (노이즈도 증가)
        # 최종 마스킹
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask)

    # Skeleton: Zhang-Suen -> 혈관 중심선과 네트워크
    # 입력: 이진 이미지
    # 출력: 이진 이미지
    def analyze(self, img_path, save_dir=r'D:\VIDEO\code\vessel_network', block_size=21, c_value=7):
        raw = cv2.imread(img_path)
        if raw is None: 
            print(f"이미지를 불러올 수 없습니다: {img_path}")
            return
        
        # 1. 정밀 ROI (붉은색 기반)
        roi_mask, hull = self.get_color_roi(raw)
        if roi_mask is None: return
        
        # 2. 혈관 및 중심선
        vessel = self.get_clean_vessels(raw, roi_mask, block_size, c_value)
        # 조건에 따라 픽셀 제거 및 유지 (더 이상 제거할 픽셀이 없을 때까지 반복)
        skeleton = skeletonize(vessel > 0).astype(np.uint8) * 255
        # 혈관의 위상 보존 (분지점, 연결성 유지)
        # 굵기 정보 제거 -> 순수한 중심선만 추출
        
        # 3. 지표 계산
        dist_map = distance_transform_edt(vessel > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(vessel > 0) / np.sum(roi_mask > 0)
        
        # 시각화 데이터 구성
        final_viz = raw.copy()
        cv2.drawContours(final_viz, [hull], -1, (255, 0, 0), 3) # 파란색 ROI
        final_viz[vessel > 0] = [0, 255, 0] # 초록색 혈관

        # 저장 추가
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Final Overlay 저장
        save_filename_overlay = f"{base_name}_{block_size}_{c_value}_overlay.png"
        save_path_overlay = os.path.join(save_dir, save_filename_overlay)
        cv2.imwrite(save_path_overlay, final_viz)
        
        # 2. ROI Mask 저장
        save_filename_roi = f"{base_name}_{block_size}_{c_value}_roi_mask.png"
        save_path_roi = os.path.join(save_dir, save_filename_roi)
        cv2.imwrite(save_path_roi, roi_mask)
        
        # 9분할 시각화 시연
        imgs = [raw, cv2.cvtColor(raw, cv2.COLOR_BGR2Lab)[:,:,1], roi_mask, vessel, skeleton, final_viz]
        titles = ['Original', 'A-Channel (Redness)', 'Refined ROI Mask', 'Vessel Map', 'Skeleton', 'Final Overlay']
        
        plt.figure(figsize=(16, 10))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(2, 3, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape)==3 else img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        report = f"Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px"
        plt.figtext(0.5, 0.05, report, ha="center", fontsize=15, bbox={"facecolor":"blue", "alpha":0.1, "pad":10})
        plt.show()

if __name__ == "__main__":
    # --- [경로 설정 수정] ---
    # 바탕 폴더 경로
    folder_path = r'D:\VIDEO\M-24\Video_00056_00052_2m30sto2m35s_tiff'
    # 파일명
    file_name = 'frame_0009.tiff'
    
    # 두 경로를 합쳐서 전체 경로 생성
    target_path = os.path.join(folder_path, file_name)
    
    analyzer = FinalVesselSystem()
    analyzer.analyze(target_path, block_size=101, c_value=3)