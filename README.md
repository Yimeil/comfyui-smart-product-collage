\# 智能产品拼接节点 - ComfyUI插件



\## 📦 功能介绍



这是一个专门用于产品图拼接的ComfyUI自定义节点，支持自动去背景、智能布局、批量处理。



\### 包含节点



1\. \*\*智能产品拼接·企业版v3.2\*\* - 单图拼接

&nbsp;  - 支持1-9张图片

&nbsp;  - 智能去背景和抠图

&nbsp;  - 统一缩放比例，保持真实大小关系

&nbsp;  - 多种布局模式



2\. \*\*智能产品拼接·批量版v1.0\*\* - 批量拼接

&nbsp;  - 支持任意数量图片

&nbsp;  - 自动分组拼接

&nbsp;  - 批量输出

&nbsp;  - 完美替代循环节点



\## 🚀 安装步骤



\### 1. 下载文件



下载以下3个文件到本地：

\- `\_\_init\_\_.py`

\- `smart\_collage\_enterprise\_v32\_scale\_fixed.py`

\- `smart\_collage\_batch.py`



\### 2. 创建插件文件夹



在ComfyUI安装目录下创建文件夹：

```

ComfyUI/custom\_nodes/smart\_product\_collage/

```



\### 3. 复制文件



将3个文件复制到刚创建的文件夹中：

```

ComfyUI/

└── custom\_nodes/

&nbsp;   └── smart\_product\_collage/

&nbsp;       ├── \_\_init\_\_.py

&nbsp;       ├── smart\_collage\_enterprise\_v32\_scale\_fixed.py

&nbsp;       └── smart\_collage\_batch.py

```



\### 4. 安装依赖（如果还没安装）



```bash

pip install torch opencv-python numpy

```



\### 5. 重启ComfyUI



完全关闭ComfyUI，然后重新启动。



\### 6. 验证安装



在ComfyUI节点搜索框中输入 "智能产品拼接"，应该能看到2个节点。



\## 📖 使用说明



\### 节点1: 单图拼接（SmartProductCollageV32）



\*\*适用场景：\*\* 拼接少量图片（1-9张）



\*\*工作流示例：\*\*

```

LoadImage (图片1)

LoadImage (图片2)

&nbsp;   ↓

SmartProductCollageV32

&nbsp;   ↓

PreviewImage

```



\*\*参数说明：\*\*

\- `images`: 批量输入

\- `image1-9`: 单独输入（两种方式二选一）

\- `layout`: 布局模式（auto/horizontal/vertical/grid）

\- `output\_width/height`: 输出尺寸

\- `spacing`: 产品间距

\- `product\_scale`: 产品缩放比例



\### 节点2: 批量拼接（SmartProductCollageBatch）



\*\*适用场景：\*\* 批量处理大量图片（100张、1000张...）



\*\*工作流示例：\*\*

```

AD\_AnyFileList (加载100张图)

&nbsp;   ↓

ImageListToImageBatch

&nbsp;   ↓

SmartProductCollageBatch

&nbsp; images\_per\_collage: 2  ← 每2张拼一次

&nbsp;   ↓

PreviewImage (显示50张拼接图)

```



\*\*参数说明：\*\*

\- `images`: 批量输入

\- `images\_per\_collage`: 每组包含几张图（2-9）

\- `skip\_empty`: 是否跳过不完整的组

\- 其他参数同单图拼接节点



\*\*使用案例：\*\*



| 输入 | images\_per\_collage | skip\_empty | 输出 |

|------|-------------------|------------|------|

| 100张 | 2 | true | 50张 |

| 99张 | 2 | true | 49张（最后1张跳过） |

| 99张 | 2 | false | 50张（最后1张单独拼接） |

| 90张 | 3 | true | 30张 |



\## 🎨 特性



\- ✅ 自动去背景

\- ✅ 智能抠图（保留配件如螺丝等）

\- ✅ 统一缩放比例（保持产品真实大小关系）

\- ✅ 多种布局模式

\- ✅ 支持文字标签

\- ✅ 白色背景

\- ✅ 批量处理



\## 🐛 故障排查



\### 节点不显示



1\. 检查文件是否在正确的位置

2\. 检查 `\_\_init\_\_.py` 文件内容

3\. 完全重启ComfyUI

4\. 查看终端错误信息



\### 运行错误



1\. 确保安装了OpenCV: `pip install opencv-python`

2\. 检查输入图片格式

3\. 查看终端日志



\### 拼接结果不理想



1\. 调整 `bg\_threshold` 参数（背景阈值）

2\. 调整 `product\_scale` 参数（产品缩放）

3\. 调整 `spacing` 参数（间距）



\## 📝 更新日志



\### v1.0 (2025-01-24)

\- ✨ 初始版本发布

\- ✨ 添加单图拼接节点

\- ✨ 添加批量拼接节点

\- 🐛 修复产品尺寸比例问题

\- 🐛 修复批量输入处理问题



\## 📄 许可证



MIT License



\## 👤 作者



AI Assistant



\## 🙏 致谢



感谢ComfyUI社区的支持！

