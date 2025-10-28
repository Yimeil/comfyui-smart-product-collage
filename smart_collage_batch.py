"""
ComfyUI批量产品拼接节点 - 内部抠图版

功能：
- 输入N张图片，自动按指定数量分组
- 内部智能抠图，保留主体和阴影效果
- 每组自动拼接成一张白底图
- 支持添加中文/数字标签（标签在每组内循环使用）
- 支持逗号分隔或换行分隔的标签输入
- 智能主次布局（大产品自动单独一侧）
- 输出所有拼接后的图片

核心特性：
- 内部抠图算法，无需外部mask
- 智能保留产品阴影，效果更自然
- 多方法融合：边缘检测、阈值分割、颜色差异
- 软边缘处理，过渡更柔和

使用场景：
- 100张图 → 每2张拼接 → 输出50张拼接图
- 90张图 → 每3张拼接 → 输出30张拼接图
- 支持为每组中的产品添加标签（如：第1个产品7pcs、第2个产品5pcs）

标签逻辑：
- 标签数量 = 每组图片数量（images_per_collage）
- 每组都使用相同的标签
- 例如：每组2张，标签为"7pcs,5pcs"或"7pcs\n5pcs"，则每组的第1张都是7pcs，第2张都是5pcs

标签输入格式：
- 逗号分隔：7pcs,5pcs,3pcs （适合程序化生成）
- 换行分隔：7pcs\n5pcs\n3pcs （适合手动输入）

版本: 2.0 (内部抠图版 - 保留阴影)
日期: 2025-01-28
更新:
1. 移除外部masks输入参数
2. 实现内部智能抠图算法
3. 保留产品主体和阴影效果
4. 多方法融合提高抠图质量
5. 软边缘处理，过渡更自然
6. 保持所有其他功能不变
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple
import math
import os
import re
from PIL import Image, ImageDraw, ImageFont



class SmartProductCollageBatch:
    """批量产品拼接节点 - 内部抠图版"""

    def __init__(self):
        self.supported_fonts = [
            "arial.ttf",
            "simhei.ttf",
            "PingFang.ttc",
            "wqy-microhei.ttc",
            "msyh.ttf"
        ]
        # 用于存储智能布局时选定的主产品索引
        self.forced_main_product_index = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 批量输入
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

    def remove_background_with_shadow(self, image: np.ndarray, crop_margin: int) -> np.ndarray:
        """
        内部抠图方法 - 保留所有物体（主体+配件）和阴影

        参数:
            image: 原始图片 [H, W, 3]
            crop_margin: 裁剪边距百分比

        返回:
            抠出的产品图片（白底，保留所有物体和阴影）[H, W, 3]
        """
        h, w = image.shape[:2]

        # 1. 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. 使用多种方法提取前景
        # 方法1: 基于边缘检测
        edges = cv2.Canny(gray, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

        # 方法2: 基于阈值（检测非白色区域）
        # 使用较低的阈值以保留阴影
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 方法3: 基于颜色差异（检测与白色背景的差异）
        # 计算每个像素与白色的差异
        white_bg = np.full_like(image, 255)
        color_diff = cv2.absdiff(image, white_bg)
        color_diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        _, color_mask = cv2.threshold(color_diff_gray, 10, 255, cv2.THRESH_BINARY)

        # 3. 合并多种方法的结果
        combined_mask = cv2.bitwise_or(thresh, color_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edges_dilated)

        # 4. 形态学操作：闭运算填充内部空洞
        kernel = np.ones((15, 15), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 5. 🆕 找到所有连通区域（保留所有物体，包括配件）
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 如果没有找到轮廓，返回原图
            return image

        # 🆕 过滤掉太小的噪点，但保留所有有效物体
        # 计算图像总面积
        total_area = h * w
        min_area_threshold = total_area * 0.0005  # 至少是图像面积的0.05%，过滤噪点

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_threshold:
                valid_contours.append(contour)

        if not valid_contours:
            # 如果没有有效轮廓，返回原图
            return image

        print(f"   🔍 检测到 {len(valid_contours)} 个物体")

        # 6. 🆕 创建包含所有物体的mask
        # 创建精细的mask（保留阴影）
        fine_mask = np.zeros((h, w), dtype=np.float32)

        # 🆕 为所有有效轮廓创建mask
        all_contours_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(all_contours_mask, valid_contours, -1, 255, -1)

        # 🚀 向量化操作：在所有轮廓区域内，保留阴影和细节
        # 阴影区域通常比白色背景暗，但比产品主体亮
        # 使用自适应的阈值来区分背景和阴影

        # 创建mask区域的布尔索引
        mask_region = all_contours_mask > 0

        # 阴影保留：240以下的都保留，240-255之间渐变
        # 使用向量化操作替代双重循环，速度提升100倍+
        fine_mask[mask_region & (gray < 240)] = 1.0
        fine_mask[mask_region & (gray >= 240)] = (255 - gray[mask_region & (gray >= 240)]) / 15.0

        # 7. 高斯模糊使边缘更自然
        fine_mask = cv2.GaussianBlur(fine_mask, (5, 5), 0)

        # 8. 扩展边界以包含柔和的阴影
        fine_mask_dilated = cv2.dilate((fine_mask * 255).astype(np.uint8),
                                        np.ones((7, 7), np.uint8), iterations=1)
        fine_mask = fine_mask_dilated.astype(np.float32) / 255.0

        # 9. 🆕 计算包含所有物体的边界框
        # 合并所有有效轮廓的边界框
        all_points = np.vstack(valid_contours)
        x, y, w_box, h_box = cv2.boundingRect(all_points)

        # 添加边距
        margin_x = int(w_box * crop_margin / 100)
        margin_y = int(h_box * crop_margin / 100)

        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w_box = min(w - x, w_box + 2 * margin_x)
        h_box = min(h - y, h_box + 2 * margin_y)

        # 裁剪图片和mask
        cropped_image = image[y:y+h_box, x:x+w_box]
        cropped_mask = fine_mask[y:y+h_box, x:x+w_box]

        # 10. 合成到白色背景
        result = np.ones_like(cropped_image) * 255
        mask_3channel = np.stack([cropped_mask, cropped_mask, cropped_mask], axis=2)
        result = cropped_image * mask_3channel + result * (1 - mask_3channel)

        return result.astype(np.uint8)

    def calculate_circularity(self, product: np.ndarray) -> float:
        """
        计算产品主体的圆度（圆形度）

        返回: 0.0-1.0 的圆度分数
            - 接近1.0: 很圆（如手镯、圆形吊坠）
            - 接近0.0: 不圆（如链条、不规则形状）
        """
        h, w = product.shape[:2]
        gray = cv2.cvtColor(product, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # 选择最大轮廓（主体）
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 100:  # 面积太小，无法判断
            return 0.0

        # 方法1: 圆形度 = 4π × 面积 / 周长²
        # 完美圆形的值为1.0，形状越不规则值越小
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return 0.0
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # 方法2: 使用最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        circle_area = np.pi * (radius ** 2)
        circle_fill_ratio = area / circle_area if circle_area > 0 else 0

        # 方法3: 轮廓到质心的距离方差（圆形的距离方差很小）
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 计算所有轮廓点到质心的距离
            distances = []
            for point in largest_contour:
                px, py = point[0]
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                distances.append(dist)

            if len(distances) > 0:
                # 归一化标准差（圆形的标准差接近0）
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                normalized_std = std_dist / mean_dist if mean_dist > 0 else 1.0
                distance_uniformity = max(0, 1.0 - normalized_std)  # 越接近1越圆
            else:
                distance_uniformity = 0.0
        else:
            distance_uniformity = 0.0

        # 综合评分（加权平均）
        final_circularity = (
            circularity * 0.4 +           # 周长面积比
            circle_fill_ratio * 0.3 +     # 外接圆填充率
            distance_uniformity * 0.3     # 距离均匀性
        )

        # 限制在0-1范围
        final_circularity = max(0.0, min(1.0, final_circularity))

        return final_circularity

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

    def calculate_product_similarity(self, product1: np.ndarray, product2: np.ndarray) -> float:
        """
        计算两个产品的相似度（基于形状而非颜色）

        返回: 0.0-1.0 的相似度分数，1.0表示完全相同
        """
        # 调整大小到相同尺寸以便比较
        h1, w1 = product1.shape[:2]
        h2, w2 = product2.shape[:2]
        target_size = (200, 200)  # 统一尺寸

        resized1 = cv2.resize(product1, target_size, interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(product2, target_size, interpolation=cv2.INTER_AREA)

        # 提取mask（用于形状比较）
        gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
        mask1 = cv2.threshold(gray1, 240, 255, cv2.THRESH_BINARY_INV)[1]
        mask2 = cv2.threshold(gray2, 240, 255, cv2.THRESH_BINARY_INV)[1]

        # 方法1: 形状相似度（Hu矩）- 40%权重
        # Hu矩是形状的7个不变特征，不受颜色、尺度、旋转影响
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_similarity = 0.0
        if contours1 and contours2:
            # 选择最大轮廓
            contour1 = max(contours1, key=cv2.contourArea)
            contour2 = max(contours2, key=cv2.contourArea)

            # 计算Hu矩
            hu1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
            hu2 = cv2.HuMoments(cv2.moments(contour2)).flatten()

            # 使用对数尺度比较（Hu矩值范围很大）
            hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
            hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)

            # 计算欧氏距离并归一化到0-1
            hu_distance = np.linalg.norm(hu1_log - hu2_log)
            shape_similarity = max(0.0, 1.0 - hu_distance / 10.0)  # 距离越小相似度越高

        # 方法2: Mask重叠度（IoU）- 35%权重
        intersection = np.logical_and(mask1 > 0, mask2 > 0)
        union = np.logical_or(mask1 > 0, mask2 > 0)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0

        # 方法3: 边缘相似度（基于Canny边缘检测）- 15%权重
        edges1 = cv2.Canny(mask1, 50, 150)
        edges2 = cv2.Canny(mask2, 50, 150)
        edge_diff = cv2.absdiff(edges1, edges2)
        edge_similarity = 1.0 - (np.sum(edge_diff > 0) / (target_size[0] * target_size[1]))

        # 方法4: 宽高比相似度 - 10%权重
        aspect1 = w1 / h1 if h1 > 0 else 1.0
        aspect2 = w2 / h2 if h2 > 0 else 1.0
        aspect_similarity = 1.0 - min(abs(aspect1 - aspect2) / max(aspect1, aspect2), 1.0)

        # 综合相似度（加权平均，侧重形状特征）
        similarity = (
            shape_similarity * 0.40 +    # Hu矩形状匹配
            iou * 0.35 +                  # Mask重叠度
            edge_similarity * 0.15 +      # 边缘相似度
            aspect_similarity * 0.10      # 宽高比
        )

        return max(0.0, min(1.0, similarity))

    def detect_duplicate_products(self, products: List[np.ndarray], threshold: float = 0.85) -> bool:
        """
        检测是否所有产品都是重复的（相似度高）

        参数:
            threshold: 相似度阈值，高于此值认为是重复产品

        返回:
            True 如果所有产品都相似（重复）
        """
        if len(products) <= 1:
            return False

        # 计算所有产品之间的相似度
        similarities = []
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                similarity = self.calculate_product_similarity(products[i], products[j])
                similarities.append(similarity)

        if not similarities:
            return False

        # 所有产品对的平均相似度
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)

        # 判断标准：平均相似度高且最低相似度也高
        is_duplicate = avg_similarity > threshold and min_similarity > (threshold - 0.1)

        if is_duplicate:
            print(f"   🔍 检测到重复产品 (平均相似度: {avg_similarity:.2f}, 最低: {min_similarity:.2f})")

        return is_duplicate

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

        if num_products == 2:
            return "horizontal"

        if num_products > 3:
            return "grid"

        # 🔧 只有数量等于3时，才检测链条来决定是否使用 adaptive_focus
        if products and layout == "auto" and num_products == 3:
            chain_products = []  # 存储所有链条产品的信息

            for idx, p in enumerate(products):
                h, w = p.shape[:2]

                # 🎯 核心逻辑：直接使用圆度判断是否为闭合环
                circularity = self.calculate_circularity(p)

                # 🔧 简化的链条特征检测
                gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
                mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]

                # 特征1: 稀疏度（链条占比小）
                total_pixels = h * w
                mask_pixels = np.sum(mask > 0)
                area_ratio = mask_pixels / total_pixels

                # 特征2: 边缘延伸（开放式链条会延伸到边缘）
                h_margin = max(1, int(h * 0.05))
                w_margin = max(1, int(w * 0.05))
                edge_mask_pixels = (
                    np.sum(mask[0:h_margin, :] > 0) +
                    np.sum(mask[-h_margin:, :] > 0) +
                    np.sum(mask[:, 0:w_margin] > 0) +
                    np.sum(mask[:, -w_margin:] > 0)
                )
                edge_touch_ratio = edge_mask_pixels / mask_pixels if mask_pixels > 0 else 0

                # 🔧 链条判定（简化版）
                is_chain = False

                # 🎯 简化判定逻辑（基于实际测试）
                # 核心发现：
                # - 闭合手镯：圆度 > 0.6
                # - 开放项链/手链：圆度 < 0.5，且稀疏（< 0.2）
                #
                # 测试数据：
                # - 产品1: 圆度0.187, 稀疏0.130 → 应该是链条
                # - 产品2: 圆度0.319, 稀疏0.140 → 应该是链条
                # - 产品3: 圆度0.377, 稀疏0.153 → 应该是链条

                # 🔧 新判定标准（更宽松）：
                # 圆度 < 0.5 AND 稀疏度 < 0.2
                if circularity < 0.5 and area_ratio < 0.2:
                    is_chain = True
                    chain_products.append({
                        'index': idx,
                        'circularity': circularity,
                        'area_ratio': area_ratio,
                        'edge_touch_ratio': edge_touch_ratio
                    })

                    print(f"   🔗 产品{idx+1}: 检测到开放链条")
                    print(f"      圆度: {circularity:.3f}, 稀疏度: {area_ratio:.3f}, 边缘延伸: {edge_touch_ratio:.3f}")
                else:
                    print(f"   ℹ️ 产品{idx+1}: 非链条 (圆度: {circularity:.3f}, 稀疏度: {area_ratio:.3f}, 边缘: {edge_touch_ratio:.3f})")

            # 如果检测到链条，选择圆度最低的作为主图
            if len(chain_products) > 0:
                print(f"\n   📐 检测到 {len(chain_products)} 个链条产品")

                if len(chain_products) == 1:
                    # 只有1个链条，直接使用adaptive_focus
                    print(f"   ➡️  使用 adaptive_focus 布局")
                    return "adaptive_focus"
                else:
                    # 有多个链条，按圆度排序，选择圆度最低的作为主图
                    chain_products.sort(key=lambda x: x['circularity'])

                    main_chain = chain_products[0]
                    self.forced_main_product_index = main_chain['index']
                    print(f"   🎯 选择产品{main_chain['index']+1}作为主图 (圆度最低: {main_chain['circularity']:.2f})")

                    for i, chain in enumerate(chain_products):
                        mark = "👑主图" if i == 0 else "副图"
                        print(f"      {mark} 产品{chain['index']+1}: 圆度={chain['circularity']:.3f}")

                    return "adaptive_focus"
            else:
                # 没有检测到链条，使用横排布局
                print(f"   ➡️  未检测到链条，使用 horizontal 布局")
                return "horizontal"

        # 默认：3个产品且无法判断时，使用横排
        return "horizontal"

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

    def calculate_label_height(self, label: str, font_size: int, product_width: int) -> Tuple[int, int]:
        """
        计算标签区域高度

        返回: (text_h, label_area_height)
        """
        # 创建临时PIL图像用于计算文字尺寸
        temp_img = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(temp_img)

        # 获取字体
        font = self.get_available_font(font_size)

        # 计算文字尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 如果文字太宽，需要缩放字体
        if text_w > product_width * 0.8:
            scale = (product_width * 0.8) / text_w
            new_font_size = max(20, int(font_size * scale))
            font = self.get_available_font(new_font_size)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_h = bbox[3] - bbox[1]

        # 计算标签区域高度
        extra_padding = max(40, int(text_h * 0.5))
        label_area_height = text_h + extra_padding

        return text_h, label_area_height

    def add_label(self, product: np.ndarray, label: str, position: str, font_size: int, margin: int,
                  unified_label_height: int = None) -> np.ndarray:
        """
        为产品添加标签

        参数:
            unified_label_height: 统一的标签区域高度（用于对齐同一组的所有产品）
        """
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

        # 🔧 修复：使用统一的标签区域高度（如果提供）
        if unified_label_height is not None:
            label_area_height = unified_label_height
        else:
            # 否则计算独立的标签高度
            extra_padding = max(40, int(text_h * 0.5))
            label_area_height = text_h + extra_padding

        # 创建新画布
        if position == "bottom":
            new_h = h + label_area_height + margin
            new_img = Image.new('RGB', (w, new_h), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            text_x = (w - text_w) // 2
            # 🔧 修复：确保文字在标签区域中垂直居中，有足够的上下空间
            text_y = h + margin + (label_area_height - text_h) // 2
        else:  # top
            new_h = h + label_area_height + margin
            new_img = Image.new('RGB', (w, new_h), (255, 255, 255))
            new_img.paste(pil_img, (0, label_area_height + margin))
            text_x = (w - text_w) // 2
            # 🔧 修复：确保文字在标签区域中垂直居中，有足够的上下空间
            text_y = (label_area_height - text_h) // 2

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

        # 🆕 如果有强制指定的主产品索引，直接使用
        if self.forced_main_product_index is not None and 0 <= self.forced_main_product_index < len(products):
            max_idx = self.forced_main_product_index
            print(f"   🎯 使用强制指定的主产品: 产品{max_idx+1}")

            # 使用完后重置
            self.forced_main_product_index = None

            main_product = products[max_idx]
            other_products = [p for i, p in enumerate(products) if i != max_idx]

            # 强制放置在上方
            adaptive_direction = "top"

        else:
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

                # 特征2: 边缘延伸检测（增强开放式链条检测）
                gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 30, 100)

                h_margin = max(1, int(h * 0.05))
                w_margin = max(1, int(w * 0.05))

                # 检测四个边缘区域的边缘密度
                top_density = np.sum(edges[0:h_margin, :]) / (h_margin * w)
                bottom_density = np.sum(edges[-h_margin:, :]) / (h_margin * w)
                left_density = np.sum(edges[:, 0:w_margin]) / (h * w_margin)
                right_density = np.sum(edges[:, -w_margin:]) / (h * w_margin)

                # 🔧 增强开放式检测：检测产品主体是否接触边缘（非白色像素）
                # 开放式链条的特征是产品本身延伸到边缘
                mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
                top_product_pixels = np.sum(mask[0:h_margin, :]) / (h_margin * w * 255)
                bottom_product_pixels = np.sum(mask[-h_margin:, :]) / (h_margin * w * 255)
                left_product_pixels = np.sum(mask[:, 0:w_margin]) / (h * w_margin * 255)
                right_product_pixels = np.sum(mask[:, -w_margin:]) / (h * w_margin * 255)

                # 边缘密度阈值
                density_threshold = 0.05
                product_threshold = 0.02  # 产品像素阈值（开放式链条特征）

                edge_touches = 0
                open_edges = 0  # 开放边缘计数

                if top_density > density_threshold:
                    edge_touches += 1
                    necklace_score += 25
                    if top_product_pixels > product_threshold:
                        open_edges += 1
                        necklace_score += 35  # 开放边缘加分更高
                if bottom_density > density_threshold:
                    edge_touches += 1
                    necklace_score += 25
                    if bottom_product_pixels > product_threshold:
                        open_edges += 1
                        necklace_score += 35
                if left_density > density_threshold:
                    edge_touches += 1
                    necklace_score += 25
                    if left_product_pixels > product_threshold:
                        open_edges += 1
                        necklace_score += 35
                if right_density > density_threshold:
                    edge_touches += 1
                    necklace_score += 25
                    if right_product_pixels > product_threshold:
                        open_edges += 1
                        necklace_score += 35

                # 🔧 增强判定：开放边缘（产品延伸到边界）是链条的强特征
                if open_edges >= 2:  # 至少2个边缘有产品延伸
                    is_necklace = True
                    necklace_score += 80
                elif edge_touches >= 2:  # 或至少2个边缘有边缘密度
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

            # 🔧 强制规则：链条/开放弧形类产品必须放上方
            if adaptive_direction == "auto":
                if product_features[0]['is_necklace']:
                    adaptive_direction = "top"
                    print(f"   📍 检测到链条/开放弧形产品，强制放置上方")
                elif canvas_w > canvas_h * 1.2:
                    adaptive_direction = "left" if len(other_products) <= 2 else "top"
                elif canvas_h > canvas_w * 1.2:
                    adaptive_direction = "top" if len(other_products) <= 2 else "left"
                else:
                    adaptive_direction = "left" if len(other_products) <= 3 else "top"

            # 🔧 即使用户指定了方向，如果检测到链条也强制改为top
            if product_features[0]['is_necklace'] and adaptive_direction != "top":
                print(f"   ⚠️  检测到链条产品，覆盖用户设置，强制放置上方")
                adaptive_direction = "top"

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

    def batch_collage(self, images, images_per_collage, layout, output_width, output_height,
                     spacing, min_spacing, outer_padding, product_scale, crop_margin,
                     skip_empty=True, labels="", label_font_size=180,
                     label_position="bottom", label_margin=40, hide_pcs_one=False, adaptive_direction="auto"):
        """
        批量拼接主函数 - 内部抠图版

        🆕 修改: 使用内部抠图，保留主体和阴影

        参数:
            images: 输入的图片batch [N, H, W, C]
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
        print("🎨 批量产品拼接节点 v2.0 (内部抠图版 - 保留阴影)")
        print("=" * 70)
        print(f"   输入图片: {batch_size}张")
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

            # 提取当前组的图片
            group_images = images[start_idx:end_idx]

            # 处理当前组的每张图片
            products = []
            for i, img_tensor in enumerate(group_images):
                try:
                    # 转换为OpenCV
                    cv2_img = self.tensor_to_cv2(img_tensor)

                    # 🆕 使用内部抠图（保留主体和阴影）
                    product = self.remove_background_with_shadow(cv2_img, crop_margin)

                    h, w = product.shape[:2]
                    print(f"   图片{i+1}: {w}x{h}px (已抠图，保留阴影)")

                    products.append(product)

                except Exception as e:
                    print(f"   ❌ 图片{i+1}处理失败: {e}")
                    continue

            if len(products) == 0:
                print(f"   ⚠️  本组没有有效产品，跳过")
                continue

            # 重置强制主产品索引（每组独立判断）
            self.forced_main_product_index = None

            # 决定布局（传入products用于智能判断）
            final_layout = self.decide_layout(len(products), layout, products)
            print(f"   布局: {final_layout}")

            # 🔧 第一步：计算统一的标签区域高度（确保同一组标签对齐）
            max_label_height = 0
            labels_info = []  # 存储每个产品的标签信息

            for i, product in enumerate(products):
                label = ""
                current_label_position = label_position
                h, w = product.shape[:2]

                if i < len(label_list):
                    label = label_list[i]

                    # 检查是否应该隐藏标签
                    if hide_pcs_one:
                        if re.match(r'^[×x]1$|^1[件套]$|^PCS:1$', label, re.IGNORECASE):
                            label = ""
                            print(f"   图片{i+1}: 标签为1，已隐藏")

                    # 如果是垂直布局，检查是否是链条产品
                    if label and final_layout == "vertical":
                        _, _, _, is_chain = self.get_product_features(product)
                        if is_chain:
                            current_label_position = "top"
                            print(f"   图片{i+1}: 检测到链条，标签位置改为top")

                labels_info.append({
                    'label': label,
                    'position': current_label_position,
                    'product_width': w
                })

                # 计算标签高度
                if label and current_label_position != "none":
                    _, label_height = self.calculate_label_height(label, label_font_size, w)
                    max_label_height = max(max_label_height, label_height)

            # 🔧 第二步：使用统一的标签高度添加标签
            final_products = []
            for i, product in enumerate(products):
                label_info = labels_info[i]
                label = label_info['label']
                current_label_position = label_info['position']

                if label and current_label_position != "none":
                    # 使用统一的标签高度
                    product = self.add_label(product, label, current_label_position,
                                           label_font_size, label_margin, max_label_height)
                    h, w = product.shape[:2]
                    print(f"   图片{i+1}: {w}x{h}px (含标签: '{label}', 位置: {current_label_position}, 统一高度: {max_label_height}px)")
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
    "SmartProductCollageBatch": "智能产品拼接·内部抠图v2.0🎨✨",
}