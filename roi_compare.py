'''
이진 분류된 ROI 이미지 기반 IOU 유사도 측정
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_shape_similarity(path1, path2):
    # 1. 이미지 로드 및 이진화
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"❌ 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        print(f"Path1: {path1}")
        print(f"Path2: {path2}")
        return

    _, mask1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

    # 2. [수치 증명] Hu Moments를 이용한 모양 비교 (위치, 크기, 회전 무관)
    # cv2.matchShapes는 Hu Moments를 비교하여 0에 가까울수록 두 모양이 같음을 나타냄
    shape_dist = cv2.matchShapes(mask1, mask2, cv2.CONTOURS_MATCH_I1, 0)

    # 3. [시각 증명] 무게 중심(Centroid)을 기준으로 두 마스크 정렬
    def get_centered_mask(mask):
        M = cv2.moments(mask)
        if M['m00'] == 0: return np.zeros((3000, 3000), dtype=np.uint8)
        
        # 무게 중심 계산
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # 흰색 영역의 bounding box 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return np.zeros((3000, 3000), dtype=np.uint8)
        
        x, y, w, h = cv2.boundingRect(contours[0])
        roi = mask[y:y+h, x:x+w]
        
        # 충분히 큰 캔버스(3000x3000) 중앙에 배치하여 위치 차이 제거
        new_size = 3000 
        centered = np.zeros((new_size, new_size), dtype=np.uint8)
        start_x = new_size // 2 - w // 2
        start_y = new_size // 2 - h // 2
        centered[start_y:start_y+h, start_x:start_x+w] = roi
        return centered

    aligned1 = get_centered_mask(mask1)
    aligned2 = get_centered_mask(mask2)

    # 4. 정렬된 상태에서의 오차 및 IoU 계산
    intersection = cv2.bitwise_and(aligned1, aligned2)
    union = cv2.bitwise_or(aligned1, aligned2)
    iou_aligned = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    diff = cv2.absdiff(aligned1, aligned2)

    # 5. 결과 출력
    print(f"--- 📊 형상 유사성 분석 보고서 ---")
    print(f"1. Shape Dissimilarity (Hu Moments): {shape_dist:.6e}")
    print(f"   (0에 수렴할수록 위치/회전에 관계없이 모양이 완벽히 일치함)")
    print(f"2. 중심점 일치 후 IoU (정렬 후 일치도): {iou_aligned:.4f}")
    
    if shape_dist < 1e-4:
        print("\n✅ 결론: 두 마스크는 위치만 다를 뿐, 모양은 거의 동일합니다.")
    else:
        print("\n⚠️ 결론: 두 마스크의 테두리 굴곡이나 비율에 미세한 왜곡이 존재합니다.")

    # 6. 시각화
    plt.figure(figsize=(14, 7))
    
    # 히트맵: 어디가 다른지 표시
    plt.subplot(121)
    plt.imshow(diff, cmap='hot')
    plt.title("Shape Difference Heatmap\n(Centered Alignment)")
    plt.colorbar(label='Pixel Difference')
    
    # 오버레이: 두 모양을 겹쳐서 비교 (하늘색/분홍색)
    plt.subplot(122)
    overlay = np.zeros((3000, 3000, 3), dtype=np.uint8)
    overlay[aligned1 > 0] = [0, 255, 255] # Cyan (Mask 1)
    overlay[aligned2 > 0] = [255, 0, 255] # Magenta (Mask 2)
    # 중앙 부분만 확대해서 보기 (1500 근처)
    plt.imshow(overlay[1000:2000, 1000:2000])
    plt.title("Shape Overlay Comparison\n(Zoomed Center)")
    
    plt.tight_layout()
    plt.show()

# --- 실행부: 업데이트된 경로 설정 ---
if __name__ == "__main__":
    # 각 이미지가 있는 폴더 경로
    dir1 = r'D:\VIDEO\code\vessel_network\Video_00056_00001_0s'
    dir2 = r'D:\VIDEO\code\vessel_network\Video_00056_00030_0s'
    
    # 파일 이름
    file1 = 'Video_00056_00001_0s_101_3_roi_mask.png'
    file2 = 'Video_00056_00030_0s_101_3_roi_mask.png'
    
    # 전체 경로 생성
    full_path1 = os.path.join(dir1, file1)
    full_path2 = os.path.join(dir2, file2)
    
    analyze_shape_similarity(full_path1, full_path2)