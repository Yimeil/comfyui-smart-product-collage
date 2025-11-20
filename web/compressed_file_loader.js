/**
 * ComfyUI 压缩文件加载器节点 - 前端扩展
 * 支持本地文件上传功能
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 在节点加载时添加文件上传功能
app.registerExtension({
    name: "Comfy.CompressedFileLoader",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CompressedFileLoader") {

            // 保存原始的 onNodeCreated 方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 查找 archive_file widget
                const archiveWidget = this.widgets.find(w => w.name === "archive_file");

                if (archiveWidget) {
                    // 创建上传按钮
                    const uploadWidget = this.addWidget("button", "上传压缩文件 📤", "upload", () => {
                        // 创建文件输入元素
                        const fileInput = document.createElement("input");
                        fileInput.type = "file";
                        fileInput.accept = ".zip,.rar,.7z";
                        fileInput.style.display = "none";
                        document.body.appendChild(fileInput);

                        fileInput.onchange = async () => {
                            const file = fileInput.files[0];
                            if (file) {
                                try {
                                    // 显示上传进度
                                    uploadWidget.name = "上传中... ⏳";
                                    app.canvas.setDirty(true);

                                    // 直接将文件复制到 input 目录
                                    const formData = new FormData();
                                    formData.append("file", file);
                                    formData.append("subfolder", "");
                                    formData.append("type", "input");
                                    formData.append("overwrite", "true");

                                    // 使用通用文件上传端点
                                    const resp = await fetch("/upload/file", {
                                        method: "POST",
                                        body: formData,
                                    });

                                    if (resp.ok) {
                                        const data = await resp.json();

                                        // 文件名可能在 data.name 或 data.filename 中
                                        const fileName = data.name || data.filename || file.name;

                                        // 更新下拉菜单的值
                                        archiveWidget.value = fileName;

                                        // 刷新下拉菜单选项 - 添加新上传的文件
                                        if (!archiveWidget.options.values.includes(fileName)) {
                                            archiveWidget.options.values.push(fileName);
                                            archiveWidget.options.values.sort();
                                        }

                                        uploadWidget.name = "上传成功! ✅";

                                        // 2秒后恢复按钮文本
                                        setTimeout(() => {
                                            uploadWidget.name = "上传压缩文件 📤";
                                            app.canvas.setDirty(true);
                                        }, 2000);

                                        console.log("✅ 文件上传成功:", fileName);
                                    } else {
                                        const errorText = await resp.text();
                                        throw new Error(`上传失败 (${resp.status}): ${errorText}`);
                                    }
                                } catch (error) {
                                    console.error("❌ 文件上传失败:", error);
                                    uploadWidget.name = "上传失败 ❌";

                                    setTimeout(() => {
                                        uploadWidget.name = "上传压缩文件 📤";
                                        app.canvas.setDirty(true);
                                    }, 2000);

                                    alert("文件上传失败: " + error.message);
                                }
                            }

                            // 清理
                            document.body.removeChild(fileInput);
                        };

                        // 触发文件选择对话框
                        fileInput.click();
                    });

                    // 调整按钮样式
                    uploadWidget.computeSize = function(width) {
                        return [width, 30];
                    };
                }

                return r;
            };
        }
    }
});
