import os
from PIL import Image

# --- [경로 설정] ---
INPUT_DIR = r"D:\VIDEO\M-24_captured\Video_00055"
OUTPUT_DIR = r"D:\VIDEO\M-24_captured\Video_00055_R_channel"

def extract_r_channel():
    # 1. 출력 폴더 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"R 채널 저장 폴더 생성 완료: {OUTPUT_DIR}")

    # 2. 해당 경로의 파일 목록 가져오기
    file_list = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.tiff')]
    
    if not file_list:
        print("해당 경로에 PNG 파일이 없습니다.")
        return

    print(f"총 {len(file_list)}개의 파일을 처리합니다.")

    for file_name in file_list:
        file_path = os.path.join(INPUT_DIR, file_name)
        
        # 3. 이미지 열기
        img = Image.open(file_path).convert("RGB")
        
        # 4. 채널 분리 (R, G, B)
        r, g, b = img.split()
        
        # 5. R 채널 저장
        # 파일명 끝에 _R을 붙여 저장합니다.
        save_name = file_name.replace(".itff", "_R.tiff")
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        # 무손실 유지하며 저장
        r.save(save_path, format='TIFF')
        print(f"처리 완료: {save_name}")

    print("-" * 30)
    print("모든 R 채널 추출 작업이 완료되었습니다.")

if __name__ == "__main__":
    extract_r_channel()