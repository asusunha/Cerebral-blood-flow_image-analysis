import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class SimpleAChannelExtractor:
    def __init__(self):
        """A채널 추출 전용 간소화 버전"""
        pass
        
    def load_tiff(self, img_path):
        """TIFF 파일 로드 (원본 품질 유지)"""
        print(f"📂 Loading TIFF: {img_path}")
        
        with Image.open(img_path) as pil_img:
            mode = pil_img.mode
            size = pil_img.size
            print(f"   - Mode: {mode}, Size: {size[0]}x{size[1]}")
            img_array = np.array(pil_img)
        
        # 그레이스케일 → RGB 변환
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # 16bit/32bit → 8bit 정규화
        if img_array.dtype in [np.uint16, np.float32, np.float64]:
            print(f"   - Converting {img_array.dtype} to 8-bit...")
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
            img_array = img_array.astype(np.uint8)
        
        # PIL RGB → OpenCV BGR
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        print(f"✅ Loaded: {img_array.shape}, dtype={img_array.dtype}")
        return img_array
    
    def extract_a_channel(self, bgr_img):
        """LAB 색공간에서 A채널 추출"""
        print("🎨 Extracting A-channel (redness)...")
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        print(f"✅ A-channel extracted: {a_channel.shape}")
        return a_channel
    
    def show_results(self, img_path, save_tiff=True, denoise=True, denoise_original=False):
        """원본과 A채널만 시각화 및 TIFF 저장"""
        # 1. 원본 로드
        original_bgr = self.load_tiff(img_path)
        if original_bgr is None:
            print("❌ Failed to load image!")
            return
        
        # 1-1. 원본 노이즈 제거 (선택적)
        if denoise_original:
            print("🧹 Denoising original image...")
            original_bgr_clean = cv2.fastNlMeansDenoisingColored(original_bgr, None, 10, 10, 7, 21)
        else:
            original_bgr_clean = original_bgr
        
        # 2. A채널 추출
        a_channel = self.extract_a_channel(original_bgr_clean)
        
        # 3. A채널 정규화
        a_channel_normalized = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX)
        a_channel_normalized = a_channel_normalized.astype(np.uint8)
        
        # 4. A채널 노이즈 제거 (선택적)
        if denoise:
            print("🧹 Applying A-channel denoising...")
            a_channel_clean = cv2.GaussianBlur(a_channel_normalized, (5, 5), 0)
        else:
            a_channel_clean = a_channel_normalized
        
        # 5. TIFF 파일 저장
        if save_tiff:
            import os
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 원본 TIFF 저장
            original_rgb = cv2.cvtColor(original_bgr_clean, cv2.COLOR_BGR2RGB)
            original_tiff = f"{base_name}_original{'_denoised' if denoise_original else ''}.tiff"
            Image.fromarray(original_rgb).save(original_tiff, compression='tiff_deflate')
            print(f"💾 Saved: {original_tiff} ({original_rgb.shape[1]}x{original_rgb.shape[0]})")
            
            # A채널 TIFF 저장
            a_channel_tiff = f"{base_name}_a_channel.tiff"
            Image.fromarray(a_channel_clean).save(a_channel_tiff, compression='tiff_deflate')
            print(f"💾 Saved: {a_channel_tiff} ({a_channel_clean.shape[1]}x{a_channel_clean.shape[0]})")
            
            # 원본 A채널도 저장 (비교용)
            if denoise:
                a_channel_raw_tiff = f"{base_name}_a_channel_raw.tiff"
                Image.fromarray(a_channel_normalized).save(a_channel_raw_tiff, compression='tiff_deflate')
                print(f"💾 Saved (raw): {a_channel_raw_tiff}")
        
        # 6. 시각화
        print("📊 Displaying results...")
        fig_cols = 4 if denoise else 3
        plt.figure(figsize=(5*fig_cols, 8))
        
        # 원본 이미지
        plt.subplot(1, fig_cols, 1)
        plt.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f'Original TIFF\n{original_bgr.shape[1]}x{original_bgr.shape[0]} px', 
                  fontsize=12)
        plt.axis('off')
        
        # 원본 노이즈 제거 (옵션)
        if denoise_original:
            plt.subplot(1, fig_cols, 2)
            plt.imshow(cv2.cvtColor(original_bgr_clean, cv2.COLOR_BGR2RGB))
            plt.title(f'Original (Denoised)\n{original_bgr_clean.shape[1]}x{original_bgr_clean.shape[0]} px', 
                      fontsize=12)
            plt.axis('off')
        
        # A채널 (원본)
        col_idx = 3 if denoise_original else 2
        if denoise:
            plt.subplot(1, fig_cols, col_idx)
            plt.imshow(a_channel_normalized, cmap='gray')
            plt.title(f'A-Channel (Raw)\n{a_channel_normalized.shape[1]}x{a_channel_normalized.shape[0]} px', 
                      fontsize=12)
            plt.axis('off')
            col_idx += 1
        
        # A채널 (노이즈 제거)
        plt.subplot(1, fig_cols, col_idx)
        plt.imshow(a_channel_clean, cmap='gray')
        plt.title(f'A-Channel {"(Denoised)" if denoise else ""}\n{a_channel_clean.shape[1]}x{a_channel_clean.shape[0]} px', 
                  fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print(f"🎯 EXTRACTION COMPLETE")
        print("="*60)
        print(f"  File Name      : {img_path}")
        print(f"  Image Size     : {original_bgr.shape[1]} x {original_bgr.shape[0]} px")
        print(f"  Original Depth : {original_bgr.dtype}")
        print(f"  A-Channel Range (raw): {a_channel.min()} ~ {a_channel.max()}")
        print(f"  A-Channel Range (normalized): {a_channel_normalized.min()} ~ {a_channel_normalized.max()}")
        print(f"  Original Denoising: {'Applied' if denoise_original else 'Disabled'}")
        print(f"  A-Channel Denoising: {'Applied' if denoise else 'Disabled'}")
        print("="*60)
        
        return original_bgr_clean, a_channel_clean

if __name__ == "__main__":
    extractor = SimpleAChannelExtractor()
    
    # 기본 파일 설정
    target_file = 'base_f00_original_final.tiff'
    
    # 파일 존재 확인
    import os
    if os.path.exists(target_file):
        print(f"▶️ Processing: {target_file}")
        # denoise_original=True 하면 원본도 노이즈 제거 (시간 좀 걸림)
        original, a_channel = extractor.show_results(target_file, denoise_original=False)
    else:
        print(f"❌ File not found: {target_file}")
        print(f"💡 Current directory: {os.getcwd()}")
        print(f"📂 Files in current directory:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.tiff', '.tif')):
                print(f"   - {f}")