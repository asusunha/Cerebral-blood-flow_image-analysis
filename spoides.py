import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

class ColorPicker:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        self.display_image = self.image.copy()
        self.colors = []
        self.color_names = ["혈관색 (1번)", "배경색 (2번)"]
        self.window_name = "색상 선택 - 클릭하여 색상 선택"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.colors) < 2:
            # BGR 색상 가져오기
            color = self.image[y, x]
            self.colors.append(color)
            
            # 선택한 위치에 원 그리기
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            
            # 색상 정보 표시
            color_name = self.color_names[len(self.colors) - 1]
            text = f"{color_name}: RGB({color[2]}, {color[1]}, {color[0]})"
            cv2.putText(self.display_image, text, (10, 30 * len(self.colors)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, self.display_image)
            
            print(f"{color_name} 선택됨: RGB({color[2]}, {color[1]}, {color[0]})")
            
            if len(self.colors) == 2:
                print("\n두 색상이 모두 선택되었습니다. 아무 키나 누르면 이진 분류를 진행합니다.")
    
    def select_colors(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_image)
        
        print("이미지에서 두 가지 색상을 클릭하여 선택하세요:")
        print("1. 혈관색 (첫 번째 클릭)")
        print("2. 배경색 (두 번째 클릭)")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.colors
    
    def calculate_threshold(self, color1, color2):
        """두 색상의 중간값 계산"""
        threshold = ((color1.astype(float) + color2.astype(float)) / 2).astype(np.uint8)
        return threshold
    
    def binary_classification(self, threshold):
        """이진 분류 수행"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 임계값 계산 (RGB의 평균)
        threshold_value = int(np.mean(threshold))
        
        print(f"\n임계값: {threshold_value}")
        print(f"임계값 RGB: ({threshold[2]}, {threshold[1]}, {threshold[0]})")
        
        # 이진화
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        return binary, threshold_value

def main():
    # 파일 경로 설정
    default_path = r"D:\VIDEO\M-24\final_roi_um2_no_crop\result_2153.12um_2049.87um_1_52\Aligned_ROI\target_f00_aligned_hybrid_97.tiff"
    
    # 파일이 존재하는지 확인
    if os.path.exists(default_path):
        image_path = default_path
        print(f"이미지 로드: {image_path}")
    else:
        # 파일 선택 대화상자
        root = Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.tiff *.tif *.png *.jpg *.jpeg")]
        )
        root.destroy()
        
        if not image_path:
            print("이미지가 선택되지 않았습니다.")
            return
    
    try:
        # ColorPicker 객체 생성
        picker = ColorPicker(image_path)
        
        # 색상 선택
        colors = picker.select_colors()
        
        if len(colors) == 2:
            # 중간 색상 계산
            threshold = picker.calculate_threshold(colors[0], colors[1])
            
            # 이진 분류 수행
            binary_result, threshold_value = picker.binary_classification(threshold)
            
            # 결과 표시
            cv2.imshow("원본 이미지", picker.image)
            cv2.imshow("이진 분류 결과", binary_result)
            
            # 결과 저장
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, "binary_classified.png")
            cv2.imwrite(output_path, binary_result)
            print(f"\n결과 저장됨: {output_path}")
            
            print("\n아무 키나 누르면 종료합니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("색상 선택이 완료되지 않았습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()