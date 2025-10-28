"""
ComfyUI智能产品拼接节点 - 企业级版本 (比例修复版)

修复内容:
- 修复横排布局中产品尺寸比例不一致的问题
- 所有产品使用统一的缩放比例，保持真实的相对大小

版本: 3.2 (比例修复版)
日期: 2025-01-24
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional
import math


class SmartProductCollageV32:
    """智能产品拼接节点 v3.2"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": ([
                    "auto",
                    "horizontal",
                    "vertical",
                    "grid",
                    "top_1_bottom_rest",
                    "left_1_right_rest",
                ], {"default": "auto"}),
                "output_width": ("INT", {"default": 1600, "min": 512, "max": 4096, "step": 64}),
                "output_height": ("INT", {"default": 1600, "min": 512, "max": 4096, "step": 64}),
                "bg_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 5}),
                "method": (["fast", "adaptive"], {"default": "fast"}),
                "spacing": ("INT", {"default": 60, "min": 0, "max": 200, "step": 10}),
                "outer_padding": ("INT", {"default": 80, "min": 0, "max": 300, "step": 10}),
                "product_scale": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 2.0, "step": 0.05}),
                "crop_margin": ("INT", {"default": 30, "min": 0, "max": 100, "step": 5}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "label1": ("STRING", {"default": ""}),
                "label2": ("STRING", {"default": ""}),
                "label3": ("STRING", {"default": ""}),
                "label4": ("STRING", {"default": ""}),
                "label5": ("STRING", {"default": ""}),
                "label6": ("STRING", {"default": ""}),
                "label7": ("STRING", {"default": ""}),
                "label8": ("STRING", {"default": ""}),
                "label9": ("STRING", {"default": ""}),
                "label_font_size": ("INT", {"default": 48, "min": 20, "max": 500}),
                "label_position": (["bottom", "top", "none"], {"default": "bottom"}),
                "label_margin": ("INT", {"default": 40, "min": 0, "max": 200}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("拼接图",)
    FUNCTION = "create_collage"
    CATEGORY = "image/smart_collage"
    
    def tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        """ComfyUI tensor → OpenCV"""
        while len(tensor.shape) == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if len(tensor.shape) != 3:
            raise ValueError(f"❌ 异常tensor shape: {tensor.shape}")
        image = tensor.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def cv2_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """OpenCV → ComfyUI tensor"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).unsqueeze(0)
    
    def remove_background(self, image: np.ndarray, method: str, threshold: int) -> Tuple[np.ndarray, List[int]]:
        """去背景"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == "adaptive":
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.ones_like(gray) * 255, [0, 0, image.shape[1], image.shape[0]]
        
        total_area = sum([cv2.contourArea(c) for c in contours])
        min_area = total_area * 0.001
        product_mask = np.zeros_like(gray)
        valid_contours = []
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(product_mask, [contour], -1, 255, -1)
                valid_contours.append(contour)
        
        if not valid_contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(product_mask, [largest], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(largest)
        else:
            all_points = np.vstack(valid_contours)
            x, y, w, h = cv2.boundingRect(all_points)
        
        return product_mask, [x, y, w, h]
    
    def extract_product(self, image: np.ndarray, mask: np.ndarray, 
                       bbox: List[int], margin: int = 30) -> np.ndarray:
        """提取产品"""
        x, y, w, h = bbox
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        cropped_img = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2]
        result = np.ones_like(cropped_img) * 255
        mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask_3ch == 255, cropped_img, result)
        return result
    
    def add_label(self, product: np.ndarray, label: str, position: str, 
                 font_size: int, margin: int) -> np.ndarray:
        """添加标签"""
        if not label or position == "none":
            return product
        
        h, w = product.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 48.0
        thickness = max(2, int(font_size / 24))
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        if position == "bottom":
            new_h = h + text_h + baseline + margin * 2
            canvas = np.ones((new_h, w, 3), dtype=np.uint8) * 255
            canvas[0:h, 0:w] = product
            text_y = h + margin + text_h
        else:
            new_h = h + text_h + baseline + margin * 2
            canvas = np.ones((new_h, w, 3), dtype=np.uint8) * 255
            canvas[text_h + baseline + margin * 2:new_h, 0:w] = product
            text_y = margin + text_h
        
        text_x = (w - text_w) // 2
        cv2.putText(canvas, label, (text_x, text_y), font, 
                   font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return canvas
    
    def decide_layout(self, num_images: int, layout_preference: str) -> str:
        """智能决定布局"""
        if layout_preference != "auto":
            return layout_preference
        if num_images <= 3:
            return "horizontal"
        elif num_images <= 6:
            return "grid"
        else:
            return "grid"
    
    def create_layout_horizontal(self, products: List[np.ndarray], canvas_w: int, 
                                canvas_h: int, spacing: int, padding: int, 
                                scale_factor: float = 1.0) -> np.ndarray:
        """
        横排布局 - 修复版
        
        关键修复: 所有产品使用统一的缩放比例，保持真实的相对大小关系
        """
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding
        total_spacing = spacing * (len(products) - 1)
        
        # ⭐ 关键修复: 计算统一的缩放比例
        # 步骤1: 计算如果所有产品并排放置，总宽度是多少
        total_original_w = sum([p.shape[1] for p in products])
        max_original_h = max([p.shape[0] for p in products])
        
        # 步骤2: 计算需要的缩放比例（考虑宽度和高度的限制）
        scale_by_width = (available_w - total_spacing) / total_original_w
        scale_by_height = available_h / max_original_h
        
        # 步骤3: 取较小的缩放比例，确保不超出画布
        unified_scale = min(scale_by_width, scale_by_height) * scale_factor
        
        print(f"   横排布局统一缩放:")
        print(f"     原始总宽: {total_original_w}px, 最大高: {max_original_h}px")
        print(f"     宽度限制缩放: {scale_by_width:.3f}")
        print(f"     高度限制缩放: {scale_by_height:.3f}")
        print(f"     最终统一缩放: {unified_scale:.3f}")
        
        # 步骤4: 使用统一缩放比例缩放所有产品
        resized_products = []
        for i, p in enumerate(products):
            h, w = p.shape[:2]
            new_w = max(1, int(w * unified_scale))
            new_h = max(1, int(h * unified_scale))
            resized = cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            resized_products.append(resized)
            print(f"     产品{i+1}: {w}x{h} → {new_w}x{new_h}")
        
        # 步骤5: 计算实际占用尺寸
        total_w = sum([img.shape[1] for img in resized_products]) + total_spacing
        max_h = max([img.shape[0] for img in resized_products])
        
        # 步骤6: 创建画布并居中放置
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        start_x = (canvas_w - total_w) // 2
        start_y = (canvas_h - max_h) // 2
        
        current_x = start_x
        for img in resized_products:
            h, w = img.shape[:2]
            y_offset = start_y + (max_h - h) // 2
            if 0 <= y_offset < canvas_h - h and 0 <= current_x < canvas_w - w:
                canvas[y_offset:y_offset+h, current_x:current_x+w] = img
            current_x += w + spacing
        
        return canvas
    
    def create_layout_vertical(self, products: List[np.ndarray], canvas_w: int, 
                              canvas_h: int, spacing: int, padding: int,
                              scale_factor: float = 1.0) -> np.ndarray:
        """竖排布局 - 统一缩放"""
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding
        total_spacing = spacing * (len(products) - 1)
        
        # 统一缩放比例
        total_original_h = sum([p.shape[0] for p in products])
        max_original_w = max([p.shape[1] for p in products])
        
        scale_by_height = (available_h - total_spacing) / total_original_h
        scale_by_width = available_w / max_original_w
        unified_scale = min(scale_by_height, scale_by_width) * scale_factor
        
        resized_products = []
        for p in products:
            h, w = p.shape[:2]
            new_w = max(1, int(w * unified_scale))
            new_h = max(1, int(h * unified_scale))
            resized_products.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4))
        
        total_h = sum([img.shape[0] for img in resized_products]) + total_spacing
        max_w = max([img.shape[1] for img in resized_products])
        
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        start_x = (canvas_w - max_w) // 2
        start_y = (canvas_h - total_h) // 2
        
        current_y = start_y
        for img in resized_products:
            h, w = img.shape[:2]
            x_offset = start_x + (max_w - w) // 2
            if 0 <= current_y < canvas_h - h and 0 <= x_offset < canvas_w - w:
                canvas[current_y:current_y+h, x_offset:x_offset+w] = img
            current_y += h + spacing
        
        return canvas
    
    def create_layout_grid(self, products: List[np.ndarray], canvas_w: int, 
                          canvas_h: int, spacing: int, padding: int,
                          scale_factor: float = 1.0) -> np.ndarray:
        """网格布局"""
        n = len(products)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding
        col_spacing = spacing * (cols - 1)
        row_spacing = spacing * (rows - 1)
        cell_w = (available_w - col_spacing) // cols
        cell_h = (available_h - row_spacing) // rows
        
        # 每个产品独立缩放到格子大小
        resized_products = []
        for p in products:
            h, w = p.shape[:2]
            scale = min(cell_w / w, cell_h / h) * scale_factor
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized_products.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4))
        
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        actual_w = cols * cell_w + col_spacing
        actual_h = rows * cell_h + row_spacing
        start_x = (canvas_w - actual_w) // 2
        start_y = (canvas_h - actual_h) // 2
        
        for idx, img in enumerate(resized_products):
            row = idx // cols
            col = idx % cols
            h, w = img.shape[:2]
            x = start_x + col * (cell_w + spacing) + (cell_w - w) // 2
            y = start_y + row * (cell_h + spacing) + (cell_h - h) // 2
            if 0 <= y < canvas_h - h and 0 <= x < canvas_w - w:
                canvas[y:y+h, x:x+w] = img
        
        return canvas
    
    def create_layout_top_1_bottom_rest(self, products: List[np.ndarray], canvas_w: int, 
                                       canvas_h: int, spacing: int, padding: int,
                                       scale_factor: float = 1.0) -> np.ndarray:
        """上1下其余布局"""
        if len(products) == 1:
            return self.create_layout_horizontal(products, canvas_w, canvas_h, spacing, padding, scale_factor)
        
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding
        top_h = int(available_h * 0.4)
        bottom_h = available_h - top_h - spacing
        
        # 上图独立缩放
        top_img = products[0]
        h, w = top_img.shape[:2]
        scale = min(int(available_w * 0.9) / w, top_h / h) * scale_factor
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        top_img = cv2.resize(top_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 下图统一缩放
        bottom_products = products[1:]
        num_bottom = len(bottom_products)
        bottom_spacing = spacing * (num_bottom - 1)
        
        total_original_w = sum([p.shape[1] for p in bottom_products])
        max_original_h = max([p.shape[0] for p in bottom_products])
        
        scale_by_width = (available_w - bottom_spacing) / total_original_w
        scale_by_height = bottom_h / max_original_h
        unified_scale = min(scale_by_width, scale_by_height) * scale_factor
        
        resized_bottom = []
        for p in bottom_products:
            h, w = p.shape[:2]
            new_w = max(1, int(w * unified_scale))
            new_h = max(1, int(h * unified_scale))
            resized_bottom.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4))
        
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        # 放置上图
        th, tw = top_img.shape[:2]
        top_x = (canvas_w - tw) // 2
        top_y = padding
        if 0 <= top_y < canvas_h - th and 0 <= top_x < canvas_w - tw:
            canvas[top_y:top_y+th, top_x:top_x+tw] = top_img
        
        # 放置下图
        total_bottom_w = sum([img.shape[1] for img in resized_bottom]) + bottom_spacing
        max_bottom_h = max([img.shape[0] for img in resized_bottom]) if resized_bottom else 0
        start_x = (canvas_w - total_bottom_w) // 2
        start_y = padding + top_h + spacing
        
        current_x = start_x
        for img in resized_bottom:
            h, w = img.shape[:2]
            y_offset = start_y + (max_bottom_h - h) // 2
            if 0 <= y_offset < canvas_h - h and 0 <= current_x < canvas_w - w:
                canvas[y_offset:y_offset+h, current_x:current_x+w] = img
            current_x += w + spacing
        
        return canvas
    
    def create_layout_left_1_right_rest(self, products: List[np.ndarray], canvas_w: int, 
                                       canvas_h: int, spacing: int, padding: int,
                                       scale_factor: float = 1.0) -> np.ndarray:
        """左1右其余布局"""
        if len(products) == 1:
            return self.create_layout_horizontal(products, canvas_w, canvas_h, spacing, padding, scale_factor)
        
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding
        left_w = int(available_w * 0.4)
        right_w = available_w - left_w - spacing
        
        # 左图独立缩放
        left_img = products[0]
        h, w = left_img.shape[:2]
        scale = min(left_w / w, int(available_h * 0.9) / h) * scale_factor
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        left_img = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 右图统一缩放
        right_products = products[1:]
        num_right = len(right_products)
        right_spacing = spacing * (num_right - 1)
        
        total_original_h = sum([p.shape[0] for p in right_products])
        max_original_w = max([p.shape[1] for p in right_products])
        
        scale_by_height = (available_h - right_spacing) / total_original_h
        scale_by_width = right_w / max_original_w
        unified_scale = min(scale_by_height, scale_by_width) * scale_factor
        
        resized_right = []
        for p in right_products:
            h, w = p.shape[:2]
            new_w = max(1, int(w * unified_scale))
            new_h = max(1, int(h * unified_scale))
            resized_right.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4))
        
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        # 放置左图
        lh, lw = left_img.shape[:2]
        left_x = padding
        left_y = (canvas_h - lh) // 2
        if 0 <= left_y < canvas_h - lh and 0 <= left_x < canvas_w - lw:
            canvas[left_y:left_y+lh, left_x:left_x+lw] = left_img
        
        # 放置右图
        total_right_h = sum([img.shape[0] for img in resized_right]) + right_spacing
        max_right_w = max([img.shape[1] for img in resized_right]) if resized_right else 0
        start_x = padding + left_w + spacing
        start_y = (canvas_h - total_right_h) // 2
        
        current_y = start_y
        for img in resized_right:
            h, w = img.shape[:2]
            x_offset = start_x + (max_right_w - w) // 2
            if 0 <= current_y < canvas_h - h and 0 <= x_offset < canvas_w - w:
                canvas[current_y:current_y+h, x_offset:x_offset+w] = img
            current_y += h + spacing
        
        return canvas
    
    def create_collage(self, layout, output_width, output_height, 
                      bg_threshold, method, spacing, outer_padding, product_scale, crop_margin,
                      images=None, image1=None, image2=None, image3=None, image4=None, 
                      image5=None, image6=None, image7=None, image8=None, image9=None,
                      label1="", label2="", label3="", label4="", 
                      label5="", label6="", label7="", label8="", label9="",
                      label_font_size=48, label_position="bottom", label_margin=40):
        """主处理函数 v3.2"""
        
        all_images = []
        
        if images is not None:
            batch_size = images.shape[0]
            for i in range(min(batch_size, 9)):
                all_images.append(images[i])
        else:
            if image1 is not None: all_images.append(image1)
            if image2 is not None: all_images.append(image2)
            if image3 is not None: all_images.append(image3)
            if image4 is not None: all_images.append(image4)
            if image5 is not None: all_images.append(image5)
            if image6 is not None: all_images.append(image6)
            if image7 is not None: all_images.append(image7)
            if image8 is not None: all_images.append(image8)
            if image9 is not None: all_images.append(image9)
        
        if len(all_images) == 0:
            raise ValueError("❌ 错误: 请提供至少一张图片")
        
        all_labels = [label1, label2, label3, label4, label5, label6, label7, label8, label9]
        num_images = len(all_images)
        
        print("\n" + "=" * 70)
        print("🎨 智能产品拼接节点 v3.2 (比例修复版)")
        print("=" * 70)
        print(f"   图片数量: {num_images}张")
        print(f"   输出尺寸: {output_width}x{output_height}px")
        print(f"   布局模式: {layout}")
        print("=" * 70)
        
        # 处理所有图片
        products = []
        for i in range(num_images):
            print(f"\n[{i+1}/{num_images}] 处理图片...")
            
            cv2_img = self.tensor_to_cv2(all_images[i])
            print(f"   原始: {cv2_img.shape[1]}x{cv2_img.shape[0]}px")
            
            mask, bbox = self.remove_background(cv2_img, method, bg_threshold)
            product = self.extract_product(cv2_img, mask, bbox, crop_margin)
            h, w = product.shape[:2]
            print(f"   抠图: {w}x{h}px")
            
            label = all_labels[i].strip()
            if label and label_position != "none":
                product = self.add_label(product, label, label_position, label_font_size, label_margin)
                print(f"   标签: '{label}'")
            
            products.append(product)
        
        # 智能布局
        final_layout = self.decide_layout(num_images, layout)
        print(f"\n📐 布局: {final_layout}")
        
        # 创建拼接图
        layout_methods = {
            "horizontal": self.create_layout_horizontal,
            "vertical": self.create_layout_vertical,
            "grid": self.create_layout_grid,
            "top_1_bottom_rest": self.create_layout_top_1_bottom_rest,
            "left_1_right_rest": self.create_layout_left_1_right_rest,
        }
        
        layout_func = layout_methods.get(final_layout, self.create_layout_grid)
        result = layout_func(products, output_width, output_height, 
                           spacing, outer_padding, product_scale)
        
        result_tensor = self.cv2_to_tensor(result)
        
        print(f"\n✅ 完成! 输出: {output_width}x{output_height}px")
        print("=" * 70 + "\n")
        
        return (result_tensor,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "SmartProductCollageV32": SmartProductCollageV32,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartProductCollageV32": "智能产品拼接·企业版v3.2✅(比例修复版)",
}