import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
import colorsys

class ProfessionalPhotoTools:
    """Professional photography tools for wedding/portrait photographers"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[PhotoTools] Initialized on {self.device}")
    
    def skin_smoothing(self, image, strength=50):
        """Professional skin smoothing for portraits"""
        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply bilateral filter for skin smoothing
        smoothed = cv2.bilateralFilter(image, 15, strength, strength)
        
        # Blend with original based on strength
        alpha = strength / 100.0
        result = cv2.addWeighted(smoothed, alpha, image, 1-alpha, 0)
        
        return result
    
    def teeth_whitening(self, image, strength=30):
        """Whiten teeth in portraits"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Target yellow tones in teeth area
        h, s, v = cv2.split(hsv)
        
        # Reduce yellow saturation
        mask = (h > 15) & (h < 35)  # Yellow range
        s[mask] *= (1 - strength/100.0)
        v[mask] *= (1 + strength/200.0)  # Slight brightness boost
        
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def eye_enhancement(self, image, strength=40):
        """Enhance eyes - brighten and add clarity"""
        # Create sharpening kernel for eyes
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * (strength / 100.0)
        
        # Apply localized sharpening
        enhanced = cv2.filter2D(image, -1, kernel)
        
        # Blend with original
        result = cv2.addWeighted(enhanced, 0.3, image, 0.7, 0)
        
        return result
    
    def wedding_color_grade(self, image, style='warm'):
        """Apply wedding-specific color grading"""
        styles = {
            'warm': {'r': 1.1, 'g': 1.05, 'b': 0.95},
            'vintage': {'r': 1.15, 'g': 1.0, 'b': 0.85},
            'dreamy': {'r': 1.05, 'g': 1.05, 'b': 1.1},
            'classic': {'r': 1.0, 'g': 1.0, 'b': 1.0},
            'film': {'r': 1.08, 'g': 1.0, 'b': 0.92}
        }
        
        if style not in styles:
            style = 'warm'
        
        factors = styles[style]
        
        # Apply color grading
        result = image.copy().astype(np.float32)
        result[:,:,2] *= factors['r']  # Red channel
        result[:,:,1] *= factors['g']  # Green channel
        result[:,:,0] *= factors['b']  # Blue channel
        
        # Add slight vignette for wedding photos
        h, w = image.shape[:2]
        center = (w//2, h//2)
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        vignette = 1 - (dist / max_dist) * 0.3
        
        # Apply vignette
        for c in range(3):
            result[:,:,c] *= vignette
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def bokeh_effect(self, image, focus_area=None, strength=50):
        """Simulate bokeh/depth of field effect"""
        if focus_area is None:
            # Default to center focus
            h, w = image.shape[:2]
            focus_area = (w//2, h//2, min(w, h)//3)
        
        cx, cy, radius = focus_area
        
        # Create mask for focus area
        h, w = image.shape[:2]
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((X - cx)**2 + (Y - cy)**2) <= radius
        
        # Apply blur to background
        blurred = cv2.GaussianBlur(image, (21, 21), strength/10)
        
        # Combine focused and blurred areas
        result = image.copy()
        result[~mask] = blurred[~mask]
        
        # Smooth transition
        kernel = cv2.getGaussianKernel(31, 10)
        kernel = kernel * kernel.T
        mask_float = mask.astype(np.float32)
        mask_smooth = cv2.filter2D(mask_float, -1, kernel)
        
        for c in range(3):
            result[:,:,c] = (image[:,:,c] * mask_smooth + 
                            blurred[:,:,c] * (1 - mask_smooth))
        
        return result.astype(np.uint8)
    
    def golden_hour_effect(self, image):
        """Apply golden hour lighting effect"""
        # Create warm overlay
        overlay = image.copy()
        overlay[:,:,0] = np.minimum(255, overlay[:,:,0] * 0.7)  # Reduce blue
        overlay[:,:,1] = np.minimum(255, overlay[:,:,1] * 1.05)  # Slight green boost
        overlay[:,:,2] = np.minimum(255, overlay[:,:,2] * 1.2)  # Boost red
        
        # Add glow effect
        glow = cv2.GaussianBlur(overlay, (51, 51), 20)
        
        # Blend with original
        result = cv2.addWeighted(image, 0.7, glow, 0.3, 0)
        
        return result
    
    def auto_exposure_correction(self, image):
        """Automatic exposure correction for poorly lit photos"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def remove_color_cast(self, image):
        """Remove unwanted color casts from photos"""
        # Calculate average color
        avg_color = image.mean(axis=(0, 1))
        
        # Find the deviation from gray
        gray_value = avg_color.mean()
        
        # Correct each channel
        result = image.copy().astype(np.float32)
        for c in range(3):
            scale = gray_value / avg_color[c]
            result[:,:,c] *= scale
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def batch_process_wedding(self, images, style='warm'):
        """Process multiple wedding photos with consistent style"""
        results = []
        
        for img in images:
            # Apply wedding processing pipeline
            processed = self.auto_exposure_correction(img)
            processed = self.remove_color_cast(processed)
            processed = self.skin_smoothing(processed, strength=40)
            processed = self.wedding_color_grade(processed, style=style)
            processed = self.golden_hour_effect(processed)
            
            results.append(processed)
        
        return results


# Test the professional tools
if __name__ == "__main__":
    print("="*60)
    print("PROFESSIONAL PHOTOGRAPHY TOOLS TEST")
    print("="*60)
    
    tools = ProfessionalPhotoTools()
    
    # Create test image
    test_img = np.ones((600, 800, 3), dtype=np.uint8) * 150
    
    # Add some features
    cv2.circle(test_img, (300, 200), 50, (100, 100, 100), -1)  # Face
    cv2.circle(test_img, (500, 200), 50, (100, 100, 100), -1)  # Face
    cv2.rectangle(test_img, (250, 400), (550, 500), (200, 200, 200), -1)  # Body
    
    print("\n[Testing Professional Features]")
    
    # Test skin smoothing
    smoothed = tools.skin_smoothing(test_img, strength=60)
    cv2.imwrite("test_skin_smooth.jpg", smoothed)
    print("- Skin smoothing: test_skin_smooth.jpg")
    
    # Test wedding color grade
    graded = tools.wedding_color_grade(test_img, style='warm')
    cv2.imwrite("test_wedding_grade.jpg", graded)
    print("- Wedding grade: test_wedding_grade.jpg")
    
    # Test bokeh effect
    bokeh = tools.bokeh_effect(test_img, focus_area=(400, 300, 200))
    cv2.imwrite("test_bokeh.jpg", bokeh)
    print("- Bokeh effect: test_bokeh.jpg")
    
    # Test golden hour
    golden = tools.golden_hour_effect(test_img)
    cv2.imwrite("test_golden_hour.jpg", golden)
    print("- Golden hour: test_golden_hour.jpg")
    
    print("\n" + "="*60)
    print("PROFESSIONAL TOOLS READY!")
    print("="*60)