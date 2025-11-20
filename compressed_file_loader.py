"""
ComfyUI 压缩文件加载器节点

功能:
- 支持本地上传 zip、rar 等压缩文件
- 自动解压文件
- 批量输出文件内容和文件名列表

版本: 2.0
日期: 2025-01-24
"""

import os
import zipfile
import tempfile
import shutil
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
import io
import folder_paths
import hashlib

# 尝试导入 rarfile (可选)
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False
    print("⚠️ rarfile 未安装, RAR 文件支持将被禁用")
    print("   安装方法: pip install rarfile")
    print("   注意: 还需要安装 unrar 工具")


class CompressedFileLoader:
    """压缩文件加载器节点 - 支持本地文件上传"""

    def __init__(self):
        self.temp_dir = None
        self.supported_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 input 目录下的所有压缩文件
        input_dir = folder_paths.get_input_directory()
        files = []

        if os.path.exists(input_dir):
            all_files = os.listdir(input_dir)
            # 筛选压缩文件
            for f in all_files:
                if f.lower().endswith(('.zip', '.rar', '.7z')):
                    files.append(f)

        return {
            "required": {
                "archive_file": (sorted(files) if files else ["请先将压缩文件放入 input 目录"],),
                "file_filter": (["all", "images_only", "non_images"], {"default": "all"}),
                "max_files": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
            },
            "optional": {
                "extract_path_filter": ("STRING", {"default": "", "multiline": False}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, archive_file):
        """验证输入文件"""
        if not archive_file:
            return "请选择或上传一个压缩文件"

        # 验证文件格式
        if not archive_file.lower().endswith(('.zip', '.rar', '.7z')):
            return "不支持的文件格式，仅支持 .zip、.rar、.7z 文件"

        return True

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("图片列表", "文件名列表", "文件路径列表", "文件数量")
    FUNCTION = "load_archive"
    CATEGORY = "image/loader"
    OUTPUT_IS_LIST = (True, True, True, False)

    @classmethod
    def IS_CHANGED(cls, archive_file, **kwargs):
        """检测文件是否变化"""
        input_dir = folder_paths.get_input_directory()
        archive_path = os.path.join(input_dir, archive_file)

        if os.path.exists(archive_path):
            return os.path.getmtime(archive_path)
        return float("nan")

    def cleanup_temp_dir(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                print(f"⚠️ 清理临时目录失败: {e}")

    def extract_archive(self, archive_path: str) -> str:
        """解压压缩文件到临时目录"""
        # 清理之前的临时目录
        self.cleanup_temp_dir()

        # 创建新的临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="comfyui_archive_")

        file_ext = os.path.splitext(archive_path)[1].lower()

        try:
            if file_ext == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                    print(f"✅ 成功解压 ZIP 文件: {len(zip_ref.namelist())} 个文件")

            elif file_ext == '.rar':
                if not RARFILE_AVAILABLE:
                    raise RuntimeError("RAR 文件支持不可用,请安装 rarfile 和 unrar")

                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(self.temp_dir)
                    print(f"✅ 成功解压 RAR 文件: {len(rar_ref.namelist())} 个文件")

            elif file_ext == '.7z':
                raise NotImplementedError("7z 文件支持尚未实现,建议使用 ZIP 或 RAR 格式")

            else:
                raise ValueError(f"不支持的压缩文件格式: {file_ext}")

            return self.temp_dir

        except Exception as e:
            self.cleanup_temp_dir()
            raise RuntimeError(f"解压文件失败: {str(e)}")

    def get_all_files(self, directory: str, path_filter: str = "") -> List[Tuple[str, str]]:
        """
        递归获取目录下的所有文件
        返回: [(完整路径, 相对路径), ...]
        """
        files = []

        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, directory)

                # 路径过滤
                if path_filter and path_filter not in rel_path:
                    continue

                files.append((full_path, rel_path))

        return files

    def is_image_file(self, filepath: str) -> bool:
        """判断是否为图片文件"""
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.supported_image_extensions

    def load_image_from_path(self, image_path: str) -> torch.Tensor:
        """从路径加载图片并转换为 ComfyUI tensor 格式"""
        try:
            img = Image.open(image_path)

            # 转换为 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 转换为 numpy array
            img_array = np.array(img).astype(np.float32) / 255.0

            # 转换为 torch tensor [H, W, C]，然后添加 batch 维度 [1, H, W, C]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            return img_tensor

        except Exception as e:
            print(f"⚠️ 加载图片失败 ({image_path}): {e}")
            # 返回一个 1x1 的黑色图片作为占位符，包含 batch 维度
            return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    def load_archive(self, archive_file: str, file_filter: str = "all",
                    max_files: int = 100, extract_path_filter: str = ""):
        """
        主处理函数 - 加载压缩文件

        参数:
            archive_file: 压缩文件名
            file_filter: 文件过滤器 (all/images_only/non_images)
            max_files: 最大文件数量
            extract_path_filter: 路径过滤字符串

        返回:
            (图片列表, 文件名列表, 文件路径列表, 文件数量)
        """

        print("\n" + "=" * 70)
        print("📦 压缩文件加载器节点 v2.0 (支持本地上传)")
        print("=" * 70)

        # 获取压缩文件完整路径
        input_dir = folder_paths.get_input_directory()

        # 处理文件名（可能包含子目录）
        if isinstance(archive_file, str):
            # 移除可能的前缀路径
            archive_file = os.path.basename(archive_file)

        archive_path = os.path.join(input_dir, archive_file)

        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"找不到压缩文件: {archive_path}\n提示: 请先上传压缩文件或将文件放入 input 目录")

        print(f"   压缩文件: {archive_file}")
        print(f"   文件大小: {os.path.getsize(archive_path) / 1024 / 1024:.2f} MB")
        print(f"   过滤模式: {file_filter}")

        # 解压文件
        print("\n📂 正在解压文件...")
        extract_dir = self.extract_archive(archive_path)

        # 获取所有文件
        all_files = self.get_all_files(extract_dir, extract_path_filter)
        print(f"   找到 {len(all_files)} 个文件")

        # 应用文件过滤
        if file_filter == "images_only":
            filtered_files = [(fp, rp) for fp, rp in all_files if self.is_image_file(fp)]
            print(f"   筛选后: {len(filtered_files)} 个图片文件")
        elif file_filter == "non_images":
            filtered_files = [(fp, rp) for fp, rp in all_files if not self.is_image_file(fp)]
            print(f"   筛选后: {len(filtered_files)} 个非图片文件")
        else:
            filtered_files = all_files

        # 限制文件数量
        if len(filtered_files) > max_files:
            print(f"⚠️ 文件数量超过限制 ({len(filtered_files)} > {max_files}), 将只处理前 {max_files} 个文件")
            filtered_files = filtered_files[:max_files]

        if len(filtered_files) == 0:
            print("❌ 没有找到符合条件的文件")
            # 返回空列表（包含 batch 维度）
            empty_img = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([empty_img], ["无文件"], [""], 0)

        # 处理文件
        images = []
        filenames = []
        filepaths = []

        print(f"\n🔄 正在处理 {len(filtered_files)} 个文件...")

        for idx, (full_path, rel_path) in enumerate(filtered_files):
            filename = os.path.basename(full_path)
            filenames.append(filename)
            filepaths.append(rel_path)

            # 如果是图片文件,加载图片
            if self.is_image_file(full_path):
                img_tensor = self.load_image_from_path(full_path)
                images.append(img_tensor)
                print(f"   [{idx+1}/{len(filtered_files)}] 📷 {filename} ({img_tensor.shape[2]}x{img_tensor.shape[1]})")
            else:
                # 非图片文件,创建占位符（包含 batch 维度）
                placeholder = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
                images.append(placeholder)
                print(f"   [{idx+1}/{len(filtered_files)}] 📄 {filename}")

        file_count = len(filtered_files)

        print(f"\n✅ 完成! 成功加载 {file_count} 个文件")
        print(f"   - 图片: {len([f for f in filtered_files if self.is_image_file(f[0])])} 张")
        print(f"   - 其他: {len([f for f in filtered_files if not self.is_image_file(f[0])])} 个")
        print("=" * 70 + "\n")

        # 注意: ComfyUI 的 OUTPUT_IS_LIST 为 True 时,会自动解包列表
        return (images, filenames, filepaths, file_count)

    def __del__(self):
        """析构函数 - 清理临时目录"""
        self.cleanup_temp_dir()


# ==================== API 路由处理 ====================

from aiohttp import web

async def upload_archive_handler(request):
    """处理压缩文件上传"""
    try:
        reader = await request.multipart()
        field = await reader.next()

        if field is None:
            return web.json_response({"error": "No file uploaded"}, status=400)

        filename = field.filename
        if not filename:
            return web.json_response({"error": "No filename provided"}, status=400)

        # 验证文件类型
        if not filename.lower().endswith(('.zip', '.rar', '.7z')):
            return web.json_response({
                "error": f"Invalid file type. Only .zip, .rar, .7z are supported. Got: {filename}"
            }, status=400)

        # 获取 input 目录并保存文件
        input_dir = folder_paths.get_input_directory()
        save_path = os.path.join(input_dir, filename)

        size = 0
        with open(save_path, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)

        print(f"✅ 文件上传成功: {filename} ({size / 1024 / 1024:.2f} MB)")

        return web.json_response({
            "success": True,
            "filename": filename,
            "size": size,
            "path": save_path
        })

    except Exception as e:
        print(f"❌ 文件上传失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def list_archives_handler(request):
    """列出所有可用的压缩文件"""
    try:
        input_dir = folder_paths.get_input_directory()
        files = []

        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.zip', '.rar', '.7z')):
                    files.append(f)

        return web.json_response({"files": sorted(files)})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "CompressedFileLoader": CompressedFileLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompressedFileLoader": "压缩文件加载器 📦",
}

# 导出 API 路由
WEB_ROUTES = [
    ("POST", "/upload/archive", upload_archive_handler),
    ("GET", "/api/archives/list", list_archives_handler),
]
