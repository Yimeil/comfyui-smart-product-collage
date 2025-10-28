"""
ComfyUI批量产品拼接节点 - 外部抠图版

功能：
- 输入N张图片和对应的masks，自动按指定数量分组
- 每组自动拼接成一张白底图
- 支持添加中文/数字标签（标签在每组内循环使用）
- 支持逗号分隔或换行分隔的标签输入
- 智能主次布局（大产品自动单独一侧）
- 输出所有拼接后的图片

变更：
- 去掉内部抠图流程 (remove_background方法)
- 新增masks输入参数
- 直接使用外部提供的masks进行抠图

使用场景：
- 100张图 + 100个mask → 每2张拼接 → 输出50张拼接图
- 90张图 + 90个mask → 每3张拼接 → 输出30张拼接图
- 支持为每组中的产品添加标签（如：第1个产品7pcs、第2个产品5pcs）

标签逻辑：
- 标签数量 = 每组图片数量（images_per_collage）
- 每组都使用相同的标签
- 例如：每组2张，标签为"7pcs,5pcs"或"7pcs\n5pcs"，则每组的第1张都是7pcs，第2张都是5pcs

标签输入格式：
- 逗号分隔：7pcs,5pcs,3pcs （适合程序化生成）
- 换行分隔：7pcs\n5pcs\n3pcs （适合手动输入）

版本: 1.8 (外部抠图版 - 修复版)
日期: 2025-01-27
更新: 
1. 去掉内部抠图流程 (remove_background方法)
2. 新增masks输入参数
3. 修改extract_product方法，直接使用外部masks
4. 修复间距、标签、链条识别和布局问题
5. 保持其他所有功能不变
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple
import math
import os
import re
import unicodedata
from PIL import Image, ImageDraw, ImageFont



class SmartProductCollageBatch:
    """批量产品拼接节点 - 外部抠图版"""
    
    def __init__(self):
        self.supported_fonts = [
            "arial.ttf", 
            "simhei.ttf", 
            "PingFang.ttc",
            "wqy-microhei.ttc", 
            "msyh.ttf"
        ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 批量输入
                "masks": ("MASK",),   # 🆕 新增masks输入
                "images_per_collage": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 9,
                    "step": 1,
                    "display": "number"
                }),
                "layout": ([
                    "auto",
                    "horizontal",
                    "vertical",
                    "grid",
                    "adaptive_focus",  # 智能主次布局：大图单独一侧
                ], {"default": "auto"}),
                "output_width": ("INT", {
                    "default": 1600,
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
                "output_height": ("INT", {
                    "default": 1600,
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
                "spacing": ("INT", {
                    "default": 60,
                    "min": 0,
                    "max": 200,
                    "step": 10
                }),
                "min_spacing": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 5,
                    "display": "number"
                }),
                "outer_padding": ("INT", {
                    "default": 80,
                    "min": 0,
                    "max": 300,
                    "step": 10
                }),
                "product_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.3,
                    "max": 2.0,
                    "step": 0.05
                }),
                "crop_margin": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 100,
                    "step": 5
                }),
            },
            "optional": {
                "skip_empty": ("BOOLEAN", {"default": True}),
                "labels": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "支持两种格式:\n1. 逗号分隔: 7pcs,5pcs,3pcs\n2. 换行分隔:\n7pcs\n5pcs\n3pcs"
                }),
                "label_font_size": ("INT", {
                    "default": 180,
                    "min": 20,
                    "max": 500,
                    "step": 10
                }),
                "label_position": (["bottom", "top", "none"], {"default": "bottom"}),
                "label_margin": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 200,
                    "step": 10
                }),
                "hide_pcs_one": ("BOOLEAN", {"default": False}),
                "adaptive_direction": (["auto", "left", "right", "top", "bottom"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("拼接图批次",)
    FUNCTION = "batch_collage"
    CATEGORY = "image/smart_collage"

    def tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor → OpenCV"""
        while len(tensor.shape) == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        image = tensor.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def cv2_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """OpenCV → Tensor"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image)
        # 添加batch维度 [H, W, 3] → [1, H, W, 3]
        return tensor.unsqueeze(0)

    def mask_tensor_to_cv2(self, mask_tensor: torch.Tensor) -> np.ndarray:
        """
        Mask Tensor → OpenCV格式
        输入: [H, W] 或 [1, H, W] 的mask tensor (0-1范围)
        输出: [H, W] 的numpy数组 (0-255范围)
        """
        # 确保是2D
        while len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor[0]

        mask = mask_tensor.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        return mask

    def extract_product_with_external_mask(self, image: np.ndarray, mask: np.ndarray, crop_margin: int) -> np.ndarray:
        """
        🆕 使用外部mask抠图（替代原来的extract_product方法）
        🔧 修复3: 改进阴影保留算法

        参数:
            image: 原始图片 [H, W, 3]
            mask: 外部提供的mask [H, W] (0-255)
            crop_margin: 裁剪边距

        返回:
            抠出的产品图片 [H, W, 3]
        """
        # 将mask标准化到0-1范围
        normalized_mask = mask.astype(np.float32) / 255.0

        # 🔧 修复3: 使用更低的阈值来保留阴影
        soft_threshold = 0.05  # 从0.1降低到0.05
        soft_mask = np.where(normalized_mask > soft_threshold, normalized_mask, 0)

        # 找到有效区域的轮廓（用于边界框计算）
        binary_mask_for_bbox = (normalized_mask > 0.2).astype(np.uint8) * 255  # 从0.3降低到0.2
        contours, _ = cv2.findContours(binary_mask_for_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 如果没有找到轮廓，返回原图
            return image

        # 计算所有轮廓的边界框
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # 添加边距
        margin_x = int(w * crop_margin / 100)
        margin_y = int(h * crop_margin / 100)

        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)

        # 裁剪图片和mask
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = soft_mask[y:y+h, x:x+w]

        # 🔧 修复3: 改进阴影保留的合成算法
        # 使用非线性变换增强阴影区域的保留度
        # 对mask应用幂函数，使低值区域（阴影）得到更高的权重
        enhanced_mask = np.power(cropped_mask, 0.5)  # 平方根变换，让0.1变成0.316，0.2变成0.447

        # 创建白色背景
        result = np.ones_like(cropped_image) * 255

        # 扩展mask到3通道
        mask_3channel = np.stack([enhanced_mask, enhanced_mask, enhanced_mask], axis=2)

        # 使用增强后的mask进行混合，更好地保留阴影
        result = cropped_image * mask_3channel + result * (1 - mask_3channel)

        return result.astype(np.uint8)

    def get_product_features(self, product: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        🔧 修复3: 改进链条识别算法 - 优先识别开放性链条

        返回: (area_ratio, aspect_ratio, edge_density, is_chain)
        """
        h, w = product.shape[:2]
        gray = cv2.cvtColor(product, cv2.COLOR_BGR2GRAY)

        # 计算非白色区域面积
        mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
        area_ratio = np.sum(mask > 0) / (h * w)

        # 宽高比
        aspect_ratio = w / h

        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (h * w)

        # 🔧 修复3: 链条特征检测 - 重点检测开放性链条
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_chain = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 检测开放边缘 - 链条通常有断开的部分
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            # 计算轮廓的紧密度（周长²/面积）
            perimeter = cv2.arcLength(largest_contour, True)
            elongation_factor = (perimeter ** 2) / contour_area if contour_area > 0 else 0

            # 检测边界接触 - 链条常常延伸到图像边缘
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            touches_edge = (x <= 5 or y <= 5 or x + w_box >= w - 5 or y + h_box >= h - 5)

            # 🔧 修复3: 开放性检测 - 检测轮廓的开放程度
            # 计算轮廓的凸性缺陷
            defects = cv2.convexityDefects(largest_contour, cv2.convexHull(largest_contour, returnPoints=False))
            major_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 1000:  # 较大的凸性缺陷表示开放性
                        major_defects += 1

            # 🔧 修复3: 链条判断条件（针对首饰类产品优化）
            chain_score = 0

            # 1. 开放性特征（最重要）
            if major_defects >= 2:  # 有多个明显的开口/凹陷
                chain_score += 4
            elif major_defects >= 1:
                chain_score += 2

            # 2. 细长特征
            if elongation_factor > 20:  # 极度细长
                chain_score += 3
            elif elongation_factor > 15:
                chain_score += 2

            # 3. 低密实度（有很多空隙）
            if solidity < 0.7:
                chain_score += 2
            elif solidity < 0.8:
                chain_score += 1

            # 4. 接触边缘
            if touches_edge:
                chain_score += 2

            # 5. 宽高比特征（首饰链条通常很长）
            if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                chain_score += 2
            elif aspect_ratio > 1.8 or aspect_ratio < 0.6:
                chain_score += 1

            # 6. 低面积但高边缘密度（细链条特征）
            if area_ratio < 0.15 and edge_density > 0.08:
                chain_score += 2
            elif area_ratio < 0.25 and edge_density > 0.06:
                chain_score += 1

            # 🔧 修复3: 降低阈值，更容易识别开放性链条
            is_chain = chain_score >= 4  # 从5降低到4，更容易识别链条

            print(f"     链条检测: 细长度={elongation_factor:.1f}, 密实度={solidity:.3f}, "
                  f"开放缺陷={major_defects}, 接触边缘={touches_edge}, 得分={chain_score}, "
                  f"判定={'开放链条' if is_chain else '非链条'}")

        return area_ratio, aspect_ratio, edge_density, is_chain

    def decide_layout(self, num_products: int, layout: str, products: List[np.ndarray] = None) -> str:
        """
        决定布局方式

        Args:
            num_products: 图片数量
            layout: 用户选择的布局模式
            products: 产品图片列表（用于智能判断）

        Returns:
            最终布局模式
        """
        if layout != "auto" and layout != "adaptive_focus":
            return layout
        # auto: 自动选择
        if num_products == 1:
            return "single"
        elif num_products == 2:
            return "horizontal"
        elif num_products >= 4:
            return "grid"
        # adaptive_focus: 智能主次布局（使用增强的链条识别）
        if layout == "auto":
            if num_products > 2 and products:
                # 🔧 使用增强的链条识别算法判断
                has_necklace = False

                for p in products:
                    h, w = p.shape[:2]
                    aspect_ratio = w / h

                    # 检测链条特征
                    if aspect_ratio > 1.3 or aspect_ratio < 0.77:
                        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 30, 100)

                        h_margin = max(1, int(h * 0.05))
                        w_margin = max(1, int(w * 0.05))

                        top_density = np.sum(edges[0:h_margin, :]) / (h_margin * w)
                        bottom_density = np.sum(edges[-h_margin:, :]) / (h_margin * w)
                        left_density = np.sum(edges[:, 0:w_margin]) / (h * w_margin)
                        right_density = np.sum(edges[:, -w_margin:]) / (h * w_margin)

                        edge_touches = sum([
                            top_density > 0.05,
                            bottom_density > 0.05,
                            left_density > 0.05,
                            right_density > 0.05
                        ])

                        if edge_touches >= 2:
                            has_necklace = True
                            break

                # 如果检测到链条，使用adaptive_focus
                if has_necklace:
                    return "adaptive_focus"

                # 否则按面积判断
                areas = [p.shape[0] * p.shape[1] for p in products]
                sorted_areas = sorted(areas, reverse=True)
                if len(sorted_areas) >= 2 and sorted_areas[0] > sorted_areas[1] * 1.5:
                    return "adaptive_focus"
            return "grid"

    def preprocess_label(self, text: str) -> str:
        """预处理标签文本：统一全角字符"""
        if not text:
            return ""

        # 替换常见的半角符号为全角
        replacements = {
            'x': '×',  # 半角x替换为乘号
            'X': '×',  # 大写X也替换
            '*': '×',  # 星号替换为乘号
        }

        result = text
        for half, full in replacements.items():
            result = result.replace(half, full)

        return result

    def get_available_font(self, size: int):
        """获取可用字体，支持中文"""
        # 常见系统字体路径
        font_paths = [
            # Windows
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/arial.ttf",
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Arial.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue

        # 使用默认字体
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def add_label(self, product: np.ndarray, label: str, position: str, font_size: int, margin: int) -> np.ndarray:
        """为产品添加标签"""
        if not label or position == "none":
            return product

        h, w = product.shape[:2]

        # 转换为PIL处理文字
        pil_img = Image.fromarray(cv2.cvtColor(product, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 获取字体
        font = self.get_available_font(font_size)

        # 计算文字尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 🔧 修复2: 更智能的字体大小调整，基于产品实际尺寸
        if text_w > w * 0.8:
            scale = (w * 0.8) / text_w
            new_font_size = max(20, int(font_size * scale))
            font = self.get_available_font(new_font_size)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        # 🔧 修复2: 确保标签区域足够大，防止被遮挡
        label_area_height = max(text_h + 20, int(h * 0.15))  # 至少是产品高度的15%

        # 创建新画布
        if position == "bottom":
            new_h = h + label_area_height + margin
            new_img = Image.new('RGB', (w, new_h), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            text_x = (w - text_w) // 2
            text_y = h + margin + (label_area_height - text_h) // 2  # 在标签区域内居中
        else:  # top
            new_h = h + label_area_height + margin
            new_img = Image.new('RGB', (w, new_h), (255, 255, 255))
            new_img.paste(pil_img, (0, label_area_height + margin))
            text_x = (w - text_w) // 2
            text_y = (label_area_height - text_h) // 2  # 在标签区域内居中

        # 绘制文字
        draw = ImageDraw.Draw(new_img)
        draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))

        # 转换回OpenCV
        return cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)

    def create_collage(self, products: List[np.ndarray], layout: str, output_width: int,
                      output_height: int, spacing: int, padding: int, scale_factor: float,
                      adaptive_direction: str = "auto", min_spacing: int = 10) -> np.ndarray:
        """创建拼接图"""
        if not products:
            return np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        if layout == "single":
            return self.create_single_layout(products[0], output_width, output_height, padding, scale_factor)
        elif layout == "horizontal":
            return self.create_horizontal_layout(products, output_width, output_height, spacing, padding, scale_factor, min_spacing)
        elif layout == "vertical":
            return self.create_vertical_layout(products, output_width, output_height, spacing, padding, scale_factor)
        elif layout == "grid":
            return self.create_grid_layout(products, output_width, output_height, spacing, padding, scale_factor)
        elif layout == "adaptive_focus":
            return self.create_adaptive_focus_layout(products, output_width, output_height, spacing, padding, scale_factor, adaptive_direction)
        else:
            return self.create_grid_layout(products, output_width, output_height, spacing, padding, scale_factor)

    def create_single_layout(self, product: np.ndarray, canvas_w: int, canvas_h: int,
                           padding: int, scale_factor: float) -> np.ndarray:
        """单个产品居中布局"""
        h, w = product.shape[:2]

        # 计算可用空间
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding

        # 计算缩放比例
        scale_x = available_w / w
        scale_y = available_h / h
        scale = min(scale_x, scale_y) * scale_factor

        # 缩放图片
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(product, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 创建画布并居中放置
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2

        if 0 <= x < canvas_w - new_w and 0 <= y < canvas_h - new_h:
            canvas[y:y+new_h, x:x+new_w] = resized

        return canvas

    def create_horizontal_layout(self, products: List[np.ndarray], canvas_w: int, canvas_h: int,
                               spacing: int, padding: int, scale_factor: float, min_spacing: int = 10) -> np.ndarray:
        """水平布局"""
        # 计算总宽度和最大高度
        total_w = sum([p.shape[1] for p in products])
        max_h = max([p.shape[0] for p in products])

        # 🔧 修复1: 减少间距，使用可调节的最小间距参数
        actual_spacing = max(spacing, min_spacing)  # 使用可调节的最小间距，更紧凑

        # 计算可用空间
        available_w = canvas_w - 2 * padding - actual_spacing * (len(products) - 1)
        available_h = canvas_h - 2 * padding

        # 计算统一缩放比例
        scale_x = available_w / total_w
        scale_y = available_h / max_h
        scale = min(scale_x, scale_y) * scale_factor

        # 缩放所有产品
        resized_products = []
        for product in products:
            h, w = product.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(product, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            resized_products.append(resized)

        # 计算实际使用的总宽度
        actual_w = sum([p.shape[1] for p in resized_products]) + actual_spacing * (len(products) - 1)
        max_resized_h = max([p.shape[0] for p in resized_products])

        # 创建画布
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 居中起始位置
        start_x = (canvas_w - actual_w) // 2
        start_y = (canvas_h - max_resized_h) // 2

        # 依次放置产品
        current_x = start_x
        for product in resized_products:
            h, w = product.shape[:2]
            y = start_y + (max_resized_h - h) // 2  # 垂直居中

            if 0 <= y < canvas_h - h and 0 <= current_x < canvas_w - w:
                canvas[y:y+h, current_x:current_x+w] = product

            current_x += w + actual_spacing

        return canvas

    def create_vertical_layout(self, products: List[np.ndarray], canvas_w: int, canvas_h: int,
                             spacing: int, padding: int, scale_factor: float) -> np.ndarray:
        """垂直布局"""
        total_h = sum([p.shape[0] for p in products])
        max_w = max([p.shape[1] for p in products])

        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding - spacing * (len(products) - 1)

        scale_x = available_w / max_w
        scale_y = available_h / total_h
        scale = min(scale_x, scale_y) * scale_factor

        resized_products = []
        for product in products:
            h, w = product.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(product, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            resized_products.append(resized)

        actual_h = sum([p.shape[0] for p in resized_products]) + spacing * (len(products) - 1)
        max_resized_w = max([p.shape[1] for p in resized_products])

        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        start_x = (canvas_w - max_resized_w) // 2
        start_y = (canvas_h - actual_h) // 2

        current_y = start_y
        for product in resized_products:
            h, w = product.shape[:2]
            x = start_x + (max_resized_w - w) // 2

            if 0 <= x < canvas_w - w and 0 <= current_y < canvas_h - h:
                canvas[current_y:current_y+h, x:x+w] = product

            current_y += h + spacing

        return canvas

    def create_grid_layout(self, products: List[np.ndarray], canvas_w: int, canvas_h: int,
                          spacing: int, padding: int, scale_factor: float) -> np.ndarray:
        """网格布局"""
        n = len(products)
        if n <= 1:
            return self.create_single_layout(products[0], canvas_w, canvas_h, padding, scale_factor)
        elif n == 2:
            return self.create_horizontal_layout(products, canvas_w, canvas_h, spacing, padding, scale_factor)
        elif n == 3:
            # 2+1布局
            rows, cols = 2, 2
        elif n == 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        elif n <= 9:
            rows, cols = 3, 3
        else:
            # 超过9个，取前9个
            products = products[:9]
            rows, cols = 3, 3

        # 计算每个格子的大小
        available_w = canvas_w - 2 * padding - spacing * (cols - 1)
        available_h = canvas_h - 2 * padding - spacing * (rows - 1)
        cell_w = available_w // cols
        cell_h = available_h // rows

        # 计算产品的统一缩放比例
        max_product_w = max([p.shape[1] for p in products])
        max_product_h = max([p.shape[0] for p in products])

        scale_x = cell_w / max_product_w
        scale_y = cell_h / max_product_h
        scale = min(scale_x, scale_y) * scale_factor

        # 创建画布
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 放置产品
        for i, product in enumerate(products):
            row = i // cols
            col = i % cols

            # 缩放产品
            h, w = product.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(product, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # 计算位置（在格子内居中）
            cell_x = padding + col * (cell_w + spacing)
            cell_y = padding + row * (cell_h + spacing)

            x = cell_x + (cell_w - new_w) // 2
            y = cell_y + (cell_h - new_h) // 2

            if 0 <= x < canvas_w - new_w and 0 <= y < canvas_h - new_h:
                canvas[y:y+new_h, x:x+new_w] = resized

        return canvas

    def create_adaptive_focus_layout(self, products: List[np.ndarray], canvas_w: int, canvas_h: int,
                                   spacing: int, padding: int, scale_factor: float, adaptive_direction: str = "auto") -> np.ndarray:
        """
        🔧 智能主次布局 - 增强链条识别（来自aaa.py）

        Args:
            direction: 主产品放置方向 (auto/left/right/top/bottom)
        """
        if len(products) < 2:
            return self.create_single_layout(products[0], canvas_w, canvas_h, padding, scale_factor)

        # 🔧 智能判断主产品 - 增强版
        product_features = []
        for i, p in enumerate(products):
            h, w = p.shape[:2]
            area = h * w
            aspect_ratio = w / h  # 宽高比

            # 🔧 增强链条检测
            is_necklace = False
            necklace_score = 0

            # 特征1: 极端宽高比（更宽松的阈值）
            if aspect_ratio > 1.3 or aspect_ratio < 0.77:  # 从1.5/0.67改为1.3/0.77
                necklace_score += 30

            # 特征2: 边缘延伸检测（降低检测阈值）
            gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)  # 从50/150降低到30/100，更敏感

            h_margin = max(1, int(h * 0.05))  # 检测边缘的范围
            w_margin = max(1, int(w * 0.05))

            # 检测四个边缘区域的边缘密度
            top_density = np.sum(edges[0:h_margin, :]) / (h_margin * w)
            bottom_density = np.sum(edges[-h_margin:, :]) / (h_margin * w)
            left_density = np.sum(edges[:, 0:w_margin]) / (h * w_margin)
            right_density = np.sum(edges[:, -w_margin:]) / (h * w_margin)

            # 边缘密度阈值（降低阈值更容易检测）
            density_threshold = 0.05  # 从0.1降低到0.05

            edge_touches = 0
            if top_density > density_threshold:
                edge_touches += 1
                necklace_score += 25
            if bottom_density > density_threshold:
                edge_touches += 1
                necklace_score += 25
            if left_density > density_threshold:
                edge_touches += 1
                necklace_score += 25
            if right_density > density_threshold:
                edge_touches += 1
                necklace_score += 25

            # 如果至少有2个边缘有延伸，判定为链条
            if edge_touches >= 2:
                is_necklace = True
                necklace_score += 50

            # 特征3: 细长度（周长/面积比）
            contours, _ = cv2.findContours(
                cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1],
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_contour, True)
                if area > 0:
                    slenderness = perimeter * perimeter / area
                    if slenderness > 50:  # 细长度高
                        necklace_score += 30
                        is_necklace = True

            product_features.append({
                'index': i,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'is_necklace': is_necklace,
                'necklace_score': necklace_score,
                'edge_touches': edge_touches,
                'score': 0
            })

        # 计算综合得分
        max_area = max([f['area'] for f in product_features])
        for feature in product_features:
            score = 0

            # 面积得分（权重降低）
            score += (feature['area'] / max_area) * 30

            # 链条得分（权重提高）
            score += feature['necklace_score']

            feature['score'] = score

        # 选择得分最高的作为主产品
        product_features.sort(key=lambda x: x['score'], reverse=True)
        max_idx = product_features[0]['index']

        # 调试信息
        print(f"   🔍 主产品识别:")
        for i, f in enumerate(product_features):
            mark = "👑主" if i == 0 else "  "
            print(f"      {mark} 产品{f['index']+1}: 得分{f['score']:.1f} "
                  f"(链条:{f['is_necklace']}, 边缘:{f['edge_touches']}, "
                  f"宽高比:{f['aspect_ratio']:.2f})")

        main_product = products[max_idx]
        other_products = [p for i, p in enumerate(products) if i != max_idx]

        # 🔧 修复：链条类产品优先放上方
        if adaptive_direction == "auto":
            if product_features[0]['is_necklace']:
                adaptive_direction = "top"
                print(f"   📍 检测到链条类产品，自动放置上方")
            elif canvas_w > canvas_h * 1.2:
                adaptive_direction = "left" if len(other_products) <= 2 else "top"
            elif canvas_h > canvas_w * 1.2:
                adaptive_direction = "top" if len(other_products) <= 2 else "left"
            else:
                adaptive_direction = "left" if len(other_products) <= 3 else "top"

        # 根据方向选择布局函数
        if adaptive_direction == "left":
            return self.create_adaptive_horizontal_split(main_product, other_products, canvas_w, canvas_h,
                                                  spacing, padding, scale_factor, main_on_left=True)
        elif adaptive_direction == "right":
            return self.create_adaptive_horizontal_split(main_product, other_products, canvas_w, canvas_h,
                                                  spacing, padding, scale_factor, main_on_left=False)
        elif adaptive_direction == "top":
            return self.create_adaptive_vertical_split(main_product, other_products, canvas_w, canvas_h,
                                                spacing, padding, scale_factor, main_on_top=True)
        elif adaptive_direction == "bottom":
            return self.create_adaptive_vertical_split(main_product, other_products, canvas_w, canvas_h,
                                                spacing, padding, scale_factor, main_on_top=False)
        else:
            return self.create_adaptive_horizontal_split(main_product, other_products, canvas_w, canvas_h,
                                                  spacing, padding, scale_factor, main_on_left=True)

    def create_adaptive_horizontal_split(self, main_product: np.ndarray, other_products: List[np.ndarray],
                                   canvas_w: int, canvas_h: int, spacing: int,
                                   padding: int, scale_factor: float, main_on_left: bool = True) -> np.ndarray:
        """
        左右分割布局（来自aaa.py）
        Args:
            main_on_left: True=主产品在左，False=主产品在右
        """
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding

        # 主产品占45%宽度
        main_w = int(available_w * 0.45)
        other_w = available_w - main_w - spacing

        # 缩放主产品
        mh, mw = main_product.shape[:2]
        main_scale = min(main_w / mw, available_h / mh) * scale_factor
        main_new_w = max(1, int(mw * main_scale))
        main_new_h = max(1, int(mh * main_scale))
        main_resized = cv2.resize(main_product, (main_new_w, main_new_h), interpolation=cv2.INTER_LANCZOS4)

        # 缩放其他产品（竖排）
        num_others = len(other_products)
        other_spacing = spacing * (num_others - 1)

        max_original_w = max([p.shape[1] for p in other_products])
        total_original_h = sum([p.shape[0] for p in other_products])

        scale_by_width = other_w / max_original_w
        scale_by_height = (available_h - other_spacing) / total_original_h
        unified_scale = min(scale_by_width, scale_by_height) * scale_factor

        others_resized = []
        for p in other_products:
            h, w = p.shape[:2]
            new_w = max(1, int(w * unified_scale))
            new_h = max(1, int(h * unified_scale))
            others_resized.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4))

        # 创建画布
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 决定位置
        if main_on_left:
            # 主产品在左
            main_x = padding
            others_start_x = padding + main_w + spacing
        else:
            # 主产品在右
            main_x = padding + other_w + spacing
            others_start_x = padding

        # 放置主产品（居中）
        main_y = (canvas_h - main_new_h) // 2
        if 0 <= main_x < canvas_w - main_new_w and 0 <= main_y < canvas_h - main_new_h:
            canvas[main_y:main_y+main_new_h, main_x:main_x+main_new_w] = main_resized

        # 放置其他产品（竖排居中）
        total_other_h = sum([img.shape[0] for img in others_resized]) + other_spacing
        max_other_w = max([img.shape[1] for img in others_resized])

        start_y = (canvas_h - total_other_h) // 2
        start_x = others_start_x + (other_w - max_other_w) // 2

        current_y = start_y
        for img in others_resized:
            h, w = img.shape[:2]
            x = start_x + (max_other_w - w) // 2
            if 0 <= x < canvas_w - w and 0 <= current_y < canvas_h - h:
                canvas[current_y:current_y+h, x:x+w] = img
            current_y += h + spacing

        return canvas

    def create_adaptive_vertical_split(self, main_product: np.ndarray, other_products: List[np.ndarray],
                                canvas_w: int, canvas_h: int, spacing: int,
                                padding: int, scale_factor: float, main_on_top: bool = True) -> np.ndarray:
        """
        上下分割布局（来自aaa.py）
        Args:
            main_on_top: True=主产品在上，False=主产品在下
        """
        available_w = canvas_w - 2 * padding
        available_h = canvas_h - 2 * padding

        # 主产品占45%高度
        main_h = int(available_h * 0.45)
        other_h = available_h - main_h - spacing

        # 缩放主产品
        mh, mw = main_product.shape[:2]
        main_scale = min(available_w / mw, main_h / mh) * scale_factor
        main_new_w = max(1, int(mw * main_scale))
        main_new_h = max(1, int(mh * main_scale))
        main_resized = cv2.resize(main_product, (main_new_w, main_new_h), interpolation=cv2.INTER_LANCZOS4)

        # 缩放其他产品（横排）
        num_others = len(other_products)
        other_spacing = spacing * (num_others - 1)

        total_original_w = sum([p.shape[1] for p in other_products])
        max_original_h = max([p.shape[0] for p in other_products])

        scale_by_width = (available_w - other_spacing) / total_original_w
        scale_by_height = other_h / max_original_h
        unified_scale = min(scale_by_width, scale_by_height) * scale_factor

        others_resized = []
        for p in other_products:
            h, w = p.shape[:2]
            new_w = max(1, int(w * unified_scale))
            new_h = max(1, int(h * unified_scale))
            others_resized.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4))

        # 创建画布
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 决定位置
        if main_on_top:
            # 主产品在上
            main_y = padding
            others_start_y = padding + main_h + spacing
        else:
            # 主产品在下
            main_y = padding + other_h + spacing
            others_start_y = padding

        # 放置主产品（居中）
        main_x = (canvas_w - main_new_w) // 2
        if 0 <= main_x < canvas_w - main_new_w and 0 <= main_y < canvas_h - main_new_h:
            canvas[main_y:main_y+main_new_h, main_x:main_x+main_new_w] = main_resized

        # 放置其他产品（横排居中）
        total_other_w = sum([img.shape[1] for img in others_resized]) + other_spacing
        max_other_h = max([img.shape[0] for img in others_resized])

        start_x = (canvas_w - total_other_w) // 2
        start_y = others_start_y + (other_h - max_other_h) // 2

        current_x = start_x
        for img in others_resized:
            h, w = img.shape[:2]
            y = start_y + (max_other_h - h) // 2
            if 0 <= y < canvas_h - h and 0 <= current_x < canvas_w - w:
                canvas[y:y+h, current_x:current_x+w] = img
            current_x += w + spacing

        return canvas

    def batch_collage(self, images, masks, images_per_collage, layout, output_width, output_height,
                     spacing, min_spacing, outer_padding, product_scale, crop_margin,
                     skip_empty=True, labels="", label_font_size=180,
                     label_position="bottom", label_margin=40, hide_pcs_one=False, adaptive_direction="auto"):
        """
        批量拼接主函数 - 外部抠图版

        🆕 修改: 新增masks参数，去掉内部抠图流程

        参数:
            images: 输入的图片batch [N, H, W, C]
            masks: 输入的mask batch [N, H, W] 🆕
            images_per_collage: 每张拼接图包含多少张原图
            labels: 标签文本，每行一个标签
            label_font_size: 标签字体大小
            label_position: 标签位置 (bottom/top/none)
            label_margin: 标签与产品的间距
            hide_pcs_one: 当标签为"×1"或"x1"时隐藏标签
            adaptive_direction: adaptive_focus布局的主产品方向
            其他参数: 拼接设置

        返回:
            拼接后的图片batch
        """

        batch_size = images.shape[0]
        mask_batch_size = masks.shape[0]

        # 检查images和masks数量是否匹配
        if batch_size != mask_batch_size:
            print(f"❌ 错误: 图片数量({batch_size})与mask数量({mask_batch_size})不匹配")
            # 返回空白图
            empty = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
            return (self.cv2_to_tensor(empty),)

        total_groups = math.ceil(batch_size / images_per_collage)

        # 解析标签 - 支持逗号和换行两种分隔方式
        label_list = []
        if labels and labels.strip():
            labels_text = self.preprocess_label(labels.strip())

            # 检测分隔符类型
            if ',' in labels_text:
                # 逗号分隔模式
                label_list = [self.preprocess_label(label.strip()) for label in labels_text.split(',') if label.strip()]
                print(f"   📝 标签数量: {len(label_list)}个 (逗号分隔)")
            else:
                # 换行分隔模式（默认）
                label_list = [self.preprocess_label(line.strip()) for line in labels_text.split('\n') if line.strip()]
                print(f"   📝 标签数量: {len(label_list)}个 (换行分隔)")


        print("\n" + "=" * 70)
        print("🎨 批量产品拼接节点 v1.8 (外部抠图版 - 修复版)")
        print("=" * 70)
        print(f"   输入图片: {batch_size}张")
        print(f"   输入masks: {mask_batch_size}张")
        print(f"   每组数量: {images_per_collage}张")
        print(f"   拼接组数: {total_groups}组")
        print(f"   输出尺寸: {output_width}x{output_height}px")
        print(f"   布局模式: {layout}")
        if layout == "adaptive_focus":
            print(f"   主产品方向: {adaptive_direction}")
        if label_list:
            print(f"   标签位置: {label_position}")
            print(f"   标签字体: {label_font_size}px")
            if hide_pcs_one:
                print(f"   隐藏PCS=1: 是")
        print("=" * 70)

        # 存储所有拼接结果
        collage_results = []

        # 按组处理
        for group_idx in range(total_groups):
            start_idx = group_idx * images_per_collage
            end_idx = min(start_idx + images_per_collage, batch_size)
            group_size = end_idx - start_idx

            print(f"\n📦 处理第{group_idx+1}/{total_groups}组 (图片{start_idx+1}-{end_idx})")

            # 跳过不完整的组（如果设置了skip_empty）
            if skip_empty and group_size < images_per_collage:
                print(f"   ⚠️  跳过不完整组 (只有{group_size}张)")
                continue

            # 提取当前组的图片和masks
            group_images = images[start_idx:end_idx]
            group_masks = masks[start_idx:end_idx]

            # 处理当前组的每张图片
            products = []
            for i, (img_tensor, mask_tensor) in enumerate(zip(group_images, group_masks)):
                try:
                    # 转换为OpenCV
                    cv2_img = self.tensor_to_cv2(img_tensor)
                    cv2_mask = self.mask_tensor_to_cv2(mask_tensor)

                    # 🆕 使用外部mask抠图（替代原来的去背景+抠图流程）
                    product = self.extract_product_with_external_mask(cv2_img, cv2_mask, crop_margin)

                    h, w = product.shape[:2]
                    print(f"   图片{i+1}: {w}x{h}px (已抠图)")

                    products.append(product)

                except Exception as e:
                    print(f"   ❌ 图片{i+1}处理失败: {e}")
                    continue

            if len(products) == 0:
                print(f"   ⚠️  本组没有有效产品，跳过")
                continue

            # 决定布局（传入products用于智能判断）
            final_layout = self.decide_layout(len(products), layout, products)
            print(f"   布局: {final_layout}")

            # 🔧 修复3: 根据布局和产品特征调整标签位置
            final_products = []
            for i, product in enumerate(products):
                # 处理标签，支持hide_pcs_one，链条产品自动使用top位置
                label = ""
                current_label_position = label_position
                if i < len(label_list):
                    label = label_list[i]

                    # 检查是否应该隐藏标签
                    if hide_pcs_one:
                        # 匹配 ×1, x1, 1件, 1套, PCS:1 等格式
                        if re.match(r'^[×x]1$|^1[件套]$|^PCS:1$', label, re.IGNORECASE):
                            label = ""  # 隐藏标签
                            print(f"   图片{i+1}: 标签为1，已隐藏")

                    # 🔧 修复3: 如果是垂直布局，检查是否是链条产品，自动改为top位置
                    if label and final_layout == "vertical":
                        # 检查当前产品是否是链条
                        _, _, _, is_chain = self.get_product_features(product)
                        if is_chain:
                            current_label_position = "top"
                            print(f"   图片{i+1}: 检测到链条，标签位置改为top")

                if label and current_label_position != "none":
                    product = self.add_label(product, label, current_label_position,
                                           label_font_size, label_margin)
                    h, w = product.shape[:2]
                    print(f"   图片{i+1}: {w}x{h}px (含标签: '{label}', 位置: {current_label_position})")
                else:
                    h, w = product.shape[:2]
                    print(f"   图片{i+1}: {w}x{h}px")

                final_products.append(product)

            if len(final_products) == 0:
                print(f"   ⚠️  本组没有有效产品，跳过")
                continue

            # 创建拼接图
            try:
                collage = self.create_collage(final_products, final_layout, output_width, output_height,
                                            spacing, outer_padding, product_scale, adaptive_direction, min_spacing)
                
                # 转换为tensor
                collage_tensor = self.cv2_to_tensor(collage)
                collage_results.append(collage_tensor)
                
                print(f"   ✅ 拼接完成")
                
            except Exception as e:
                print(f"   ❌ 拼接失败: {e}")
                continue
        
        if len(collage_results) == 0:
            print(f"\n❌ 没有生成任何拼接图")
            # 返回空白图
            empty = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
            return (self.cv2_to_tensor(empty),)
        
        # 合并所有拼接结果
        final_batch = torch.cat(collage_results, dim=0)
        
        print(f"\n✅ 批量拼接完成!")
        print(f"   输出: {final_batch.shape[0]}张拼接图")
        print("=" * 70 + "\n")
        
        return (final_batch,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "SmartProductCollageBatch": SmartProductCollageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartProductCollageBatch": "智能产品拼接·外部抠图版v1.8🔁✨",
}