"""
智能产品拼接节点 - ComfyUI插件

包含:
1. SmartProductCollageV32 - 单图拼接节点 (支持1-9张图)
2. SmartProductCollageBatch - 批量拼接节点 (支持任意数量图片)
3. CompressedFileLoader - 压缩文件加载器节点 (支持zip、rar等格式)

作者: AI Assistant
版本: 1.1
日期: 2025-01-24
"""

# 导入节点类
from .smart_collage_enterprise import SmartProductCollageV32
from .smart_collage_batch import SmartProductCollageBatch
from .compressed_file_loader import CompressedFileLoader

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "SmartProductCollageV32": SmartProductCollageV32,
    "SmartProductCollageBatch": SmartProductCollageBatch,
    "CompressedFileLoader": CompressedFileLoader,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartProductCollageV32": "智能产品拼接·企业版v3.2✅",
    "SmartProductCollageBatch": "智能产品拼接·批量版v1.0🔁",
    "CompressedFileLoader": "压缩文件加载器 📦",
}

# 导出（ComfyUI需要）
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ 智能产品拼接节点已加载")
print("   - SmartProductCollageV32 (单图拼接)")
print("   - SmartProductCollageBatch (批量拼接)")
print("   - CompressedFileLoader (压缩文件加载器)")