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

                                    // 上传文件到服务器
                                    const formData = new FormData();
                                    formData.append("image", file);
                                    formData.append("subfolder", "");
                                    formData.append("type", "input");
                                    formData.append("overwrite", "true");

                                    const resp = await api.fetchApi("/upload/image", {
                                        method: "POST",
                                        body: formData,
                                    });

                                    if (resp.status === 200) {
                                        const data = await resp.json();

                                        // 更新下拉菜单的值
                                        archiveWidget.value = data.name;

                                        // 刷新下拉菜单选项 - 添加新上传的文件
                                        if (!archiveWidget.options.values.includes(data.name)) {
                                            archiveWidget.options.values.push(data.name);
                                            archiveWidget.options.values.sort();
                                        }

                                        uploadWidget.name = "上传成功! ✅";

                                        // 2秒后恢复按钮文本
                                        setTimeout(() => {
                                            uploadWidget.name = "上传压缩文件 📤";
                                            app.canvas.setDirty(true);
                                        }, 2000);

                                        console.log("✅ 文件上传成功:", data.name);
                                    } else {
                                        throw new Error("上传失败");
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
