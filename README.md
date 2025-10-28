# 智能产品拼接节点 - ComfyUI插件

## 📦 功能介绍

这是一个专门用于产品图拼接的ComfyUI自定义节点，支持**智能抠图、自动布局、批量处理、标签添加**。

### ✨ 核心特性

- 🎯 **智能抠图** - 内部算法，保留产品主体和阴影，无需外部mask
- 🧠 **智能布局** - 自动识别链条类产品，选择最佳布局方式
- 📝 **标签支持** - 支持中文/数字标签，自动对齐
- 🔄 **批量处理** - 支持任意数量图片，自动分组拼接
- 📐 **真实比例** - 统一缩放，保持产品真实大小关系
- 🎨 **白色背景** - 专业电商风格

### 包含节点

1. **智能产品拼接·内部抠图v2.0🎨✨** (SmartProductCollageBatch)
   - ✅ 批量处理（支持任意数量图片）
   - ✅ 内部智能抠图，保留阴影
   - ✅ 多种布局模式（auto/horizontal/vertical/grid/adaptive_focus）
   - ✅ 智能链条识别（3个产品时自动优化布局）
   - ✅ 标签支持（中文/数字，自动对齐）
   - ✅ 自动分组拼接

2. **智能产品拼接·企业版v3.2✅** (SmartProductCollageV32)
   - ✅ 单次拼接（1-9张图）
   - ✅ 简单易用

## 🚀 安装步骤

### 方法一：直接复制文件（推荐）

1. **下载文件**（共3个）：
   - `__init__.py`
   - `smart_collage_enterprise.py`
   - `smart_collage_batch.py`

2. **创建插件文件夹**：
   ```
   ComfyUI/custom_nodes/smart_product_collage/
   ```

3. **复制文件**到该文件夹：
   ```
   ComfyUI/
   └── custom_nodes/
       └── smart_product_collage/
           ├── __init__.py
           ├── smart_collage_enterprise.py
           └── smart_collage_batch.py
   ```

4. **重启ComfyUI**（完全关闭后重新启动）

5. **验证安装**：
   在节点搜索框输入 "智能产品拼接"，应该看到2个节点

### 方法二：Git克隆

```bash
cd ComfyUI/custom_nodes/
git clone <repository-url> smart_product_collage
# 重启ComfyUI
```

## 📖 使用说明

### 节点：智能产品拼接·内部抠图v2.0 (推荐)

**适用场景：** 所有产品拼接需求

#### 基础参数

| 参数 | 说明 | 默认值 | 范围 |
|-----|------|-------|------|
| `images` | 批量输入图片 | 必需 | - |
| `images_per_collage` | 每组包含几张图 | 2 | 1-9 |
| `layout` | 布局模式 | auto | 见下表 |
| `output_width` | 输出宽度 | 1600 | 512-4096 |
| `output_height` | 输出高度 | 1600 | 512-4096 |
| `spacing` | 产品间距 | 60 | 0-200 |
| `min_spacing` | 最小间距 | 10 | 0-50 |
| `outer_padding` | 外边距 | 80 | 0-300 |
| `product_scale` | 产品缩放 | 1.0 | 0.3-2.0 |
| `crop_margin` | 裁剪边距 | 30 | 0-100 |

#### 布局模式说明

| 模式 | 说明 | 适用场景 |
|-----|------|---------|
| **auto** | 🌟 智能自动选择 | **推荐**，自动根据数量和产品类型选择 |
| `horizontal` | 横排布局 | 2-3个产品 |
| `vertical` | 竖排布局 | 2-3个产品 |
| `grid` | 网格布局 | 4+个产品 |
| `adaptive_focus` | 智能主次布局 | 突出主产品（链条/项链自动识别） |

#### Auto布局自动规则

| 数量 | 自动布局 | 说明 |
|-----|---------|------|
| 1个 | 单个居中 | - |
| 2个 | 横排 | - |
| 3个 | **智能判断** | 检测到链条→主次布局，否则→横排 |
| 4+个 | 网格 | - |

**链条识别**（数量=3时生效）：
- 自动识别开放式项链/手链
- 选择圆度最低的作为主图，单独放在上方
- 其他产品横排显示在下方

#### 标签参数（可选）

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `labels` | 标签文本 | "" |
| `label_font_size` | 字体大小 | 180 |
| `label_position` | 位置 | bottom |
| `label_margin` | 标签边距 | 40 |
| `hide_pcs_one` | 隐藏"×1" | false |

**标签输入格式**：
```
# 逗号分隔（程序化）
7pcs,5pcs,3pcs

# 换行分隔（手动输入）
7pcs
5pcs
3pcs
```

**标签逻辑**：
- 标签数量 = `images_per_collage`
- 每组使用相同标签
- 例如：每组2张，标签 "7pcs,5pcs"，则每组第1张7pcs，第2张5pcs

#### 其他参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `skip_empty` | 跳过不完整组 | true |
| `adaptive_direction` | 主产品方向 | auto |

## 🎯 使用案例

### 案例1：简单批量拼接

**需求**：100张产品图，每2张拼成一张

```
AD_AnyFileList (加载100张图)
  ↓
ImageListToImageBatch
  ↓
SmartProductCollageBatch
  - images_per_collage: 2
  - layout: auto
  ↓
PreviewImage (输出50张拼接图)
```

### 案例2：带标签的拼接

**需求**：90张产品图，每3张拼一次，添加数量标签

```
ImageListToImageBatch
  ↓
SmartProductCollageBatch
  - images_per_collage: 3
  - layout: auto
  - labels: "7pcs\n5pcs\n3pcs"
  - label_position: bottom
  ↓
PreviewImage (输出30张拼接图)
```

### 案例3：项链拼接（自动识别）

**需求**：3个首饰产品（含项链），自动识别并优化布局

```
LoadImage × 3 (项链、手镯、手链)
  ↓
SmartProductCollageBatch
  - images_per_collage: 3
  - layout: auto  ← 自动识别项链作为主图
  ↓
PreviewImage
```

**自动处理**：
- 检测到项链（圆度低，开放弧形）
- 项链单独放上方（主图）
- 手镯、手链横排显示下方

### 案例4：自定义布局

**需求**：4个产品，网格布局，无标签

```
ImageListToImageBatch
  ↓
SmartProductCollageBatch
  - images_per_collage: 4
  - layout: grid
  - spacing: 80
  - product_scale: 0.9
  ↓
PreviewImage
```

## 📊 批量处理示例

| 输入 | images_per_collage | skip_empty | 输出 | 说明 |
|------|-------------------|------------|------|------|
| 100张 | 2 | true | 50张 | 完美分组 |
| 99张 | 2 | true | 49张 | 最后1张跳过 |
| 99张 | 2 | false | 50张 | 最后1张单独拼接 |
| 90张 | 3 | true | 30张 | 完美分组 |
| 100张 | 3 | true | 33张 | 最后1张跳过（99÷3） |

## 🔧 参数调整建议

### 产品间距太大/太小

```
spacing: 60      # 调整产品间距（0-200）
min_spacing: 10  # 横排布局的最小间距
outer_padding: 80  # 调整外边距
```

### 产品尺寸不合适

```
product_scale: 1.0  # 整体缩放（0.3-2.0）
crop_margin: 30     # 裁剪边距（0-100）
```

### 标签位置/大小不对

```
label_font_size: 180  # 字体大小（20-500）
label_position: bottom  # 位置（bottom/top/none）
label_margin: 40      # 标签与产品间距
```

### 抠图效果不理想

当前版本使用内部智能抠图，自动处理。如果效果不好：
1. 确保图片背景尽量接近白色
2. 调整 `crop_margin` 参数

## 🐛 故障排查

### 节点不显示

1. ✅ 检查文件位置：`ComfyUI/custom_nodes/smart_product_collage/`
2. ✅ 确认3个文件都在文件夹内
3. ✅ 完全重启ComfyUI（关闭所有窗口）
4. ✅ 查看终端错误信息

### 运行错误

1. **缺少依赖**：
   ```bash
   pip install opencv-python numpy torch pillow
   ```

2. **导入错误**：
   检查 `__init__.py` 中的导入语句是否正确

3. **内存不足**：
   - 减少 `images_per_collage`
   - 降低 `output_width/height`

### 链条识别不准确

当前使用圆度+稀疏度判断：
- 闭合手镯/手链：圆度高，居中
- 开放项链：圆度低，稀疏

如果识别错误：
- 调整为手动布局：`layout: adaptive_focus`
- 或使用：`layout: horizontal` / `layout: grid`

### 标签显示异常

1. **标签数量**必须等于 `images_per_collage`
2. 检查标签格式（逗号或换行分隔）
3. 调整 `label_font_size` 和 `label_margin`

## 📝 更新日志

### v2.0 (2025-01-28)
- ✨ 重大更新：内部智能抠图（无需外部mask）
- ✨ 智能链条识别（3个产品时自动优化布局）
- ✨ 标签支持（中文/数字，自动对齐）
- ✨ 新增圆度检测算法
- ✨ 新增 `auto` 布局模式
- ✨ 保留产品阴影效果
- 🐛 修复批量处理的内存问题

### v1.0 (2025-01-24)
- ✨ 初始版本发布
- ✨ 添加单图拼接节点
- ✨ 添加批量拼接节点
- 🐛 修复产品尺寸比例问题

## 💡 使用技巧

1. **优先使用 `layout: auto`**，智能判断最佳布局
2. **标签自动对齐**，同组产品标签高度统一
3. **链条自动识别**，3个产品时自动优化
4. **批量处理大文件**，建议 `skip_empty: true` 避免不完整组
5. **调整间距优先级**：`min_spacing` → `spacing` → `outer_padding`

## 📄 许可证

MIT License

## 👤 作者

AI Assistant

## 🙏 致谢

感谢ComfyUI社区的支持！

---

**关键词**：ComfyUI, 产品拼接, 智能抠图, 批量处理, 电商图片, 项链识别, 自动布局
