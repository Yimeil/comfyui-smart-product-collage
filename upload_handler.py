"""
自定义文件上传处理器
用于处理压缩文件的上传
"""

import os
import folder_paths
from aiohttp import web
import mimetypes

class UploadHandler:
    """处理压缩文件上传的 API"""

    @staticmethod
    async def upload_archive(request):
        """
        处理压缩文件上传
        接受 multipart/form-data 格式的文件上传
        """
        try:
            reader = await request.multipart()
            field = await reader.next()

            if field is None:
                return web.json_response({"error": "No file uploaded"}, status=400)

            # 获取文件名
            filename = field.filename
            if not filename:
                return web.json_response({"error": "No filename provided"}, status=400)

            # 验证文件类型
            if not filename.lower().endswith(('.zip', '.rar', '.7z')):
                return web.json_response({
                    "error": f"Invalid file type. Only .zip, .rar, .7z are supported. Got: {filename}"
                }, status=400)

            # 获取 input 目录
            input_dir = folder_paths.get_input_directory()

            # 构建保存路径
            save_path = os.path.join(input_dir, filename)

            # 保存文件
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

    @staticmethod
    async def list_archives(request):
        """列出所有可用的压缩文件"""
        try:
            input_dir = folder_paths.get_input_directory()
            files = []

            if os.path.exists(input_dir):
                for f in os.listdir(input_dir):
                    if f.lower().endswith(('.zip', '.rar', '.7z')):
                        files.append(f)

            return web.json_response({
                "files": sorted(files)
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


# 导出路由配置
def setup_routes(routes):
    """设置路由"""
    routes.append(web.post('/upload/archive', UploadHandler.upload_archive))
    routes.append(web.get('/api/archives/list', UploadHandler.list_archives))
    print("✅ 压缩文件上传 API 已注册")
    print("   - POST /upload/archive - 上传压缩文件")
    print("   - GET /api/archives/list - 列出压缩文件")
