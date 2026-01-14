import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure, morphology

class FullyAutomatedVesselExtractor:
    def __init__(self, min_object_size=50):
        # 너무 작은 노이즈 점들을 제거할 기준 크기
        self.min_object_size = min_object_size

    def process(self, image_path):
        # 1. 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            print(f"오류: 파일을 찾을 수 없습니다. 경로: {image_path}")
            return None

        # 2. 그린 채널 추출 (혈관 대비가 가장 높음)
        green_ch = img[:, :, 1]

        # 3. 배경 평탄화 (조명 불균형 제거)
        # 큼직한 가우시안 블러로 배경 이미지를 만든 뒤 원본에서 뺍니다.
        bg = cv2.GaussianBlur(green_ch, (51, 51), 0)
        flat_img = cv2.addWeighted(green_ch, 1, bg, -1, 128)

        # 4. Sato 필터 적용 (핵심: 혈관 같은 선형 구조물 강조)
        # sigmas=[1, 2, 3]은 다양한 두께의 혈관을 찾도록 설정한 것입니다.
        vessels = sato(flat_img, sigmas=range(1, 4), black_ridges=True)

        # 5. 결과 정규화 (0~255 범위로 변환)
        vessels_rescaled = exposure.rescale_intensity(vessels, out_range=(0, 255)).astype(np.uint8)

        # 6. 자동 이진화 (Otsu 알고리즘)
        # 배경과 혈관을 가르는 최적의 임계값을 자동으로 찾습니다.
        _, binary = cv2.threshold(vessels_rescaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. 후처리 (노이즈 제거)
        # 불필요한 작은 점 제거 및 미세한 구멍 채우기
        bool_binary = binary.astype(bool)
        clean_bool = morphology.remove_small_objects(bool_binary, min_size=self.min_object_size)
        clean_bool = morphology.remove_small_holes(clean_bool, area_threshold=self.min_object_size)
        
        final_binary = (clean_bool * 255).astype(np.uint8)

        return img, final_binary

    def save_and_show(self, original, segmented, output_path='result.png'):
        # 결과 저장
        cv2.imwrite(output_path, segmented)
        
        # 화면 출력을 위해 크기 조정 (모니터보다 클 경우 대비)
        h, w = original.shape[:2]
        display_w = 800
        display_h = int(h * (display_w / w))
        
        combined = np.hstack((cv2.resize(original, (display_w, display_h)), 
                              cv2.cvtColor(cv2.resize(segmented, (display_w, display_h)), cv2.COLOR_GRAY2BGR)))
        
        cv2.imshow('Left: Original | Right: Automated Segmentation', combined)
        print(f"결과가 저장되었습니다: {output_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    extractor = FullyAutomatedVesselExtractor(min_object_size=100)
    
    # 사용자님의 이미지 경로
    img_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Baseline_ROI\base_f00_original.tiff"
    
    if os.path.exists(img_path):
        ori, seg = extractor.process(img_path)
        if seg is not None:
            extractor.save_and_show(ori, seg, 'vessel_extraction_result.png')
    else:
        print("파일 경로를 다시 확인해주세요.")