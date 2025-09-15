/**
 * 为FileListDownloader和BatchVideoDownloader添加下载按钮功能
 */

console.log("🚀 BatchVideoProcessor 下载处理扩展开始加载...");

import { app } from "/scripts/app.js";

// 下载文件的函数
function downloadFile(filename, type = "output", subfolder = "") {
    // 动态获取当前访问的地址构造下载URL
    const baseUrl = `${window.location.protocol}//${window.location.host}/view`;
    let downloadUrl = `${baseUrl}?filename=${encodeURIComponent(filename)}&type=${type}`;
    
    if (subfolder) {
        downloadUrl += `&subfolder=${encodeURIComponent(subfolder)}`;
    }
    
    console.log(`🔗 构造下载链接: ${downloadUrl}`);
    
    // 直接打开下载链接
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    console.log(`✅ 开始下载: ${filename}`);
}

app.registerExtension({
    name: "BatchVideoProcessor.DownloadButton",
    
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log(`🔍 检查节点: ${nodeData.name}`);
        
        // 只处理我们的下载节点
        if (nodeData.name !== "FileListDownloader" && nodeData.name !== "BatchVideoDownloader") {
            return;
        }
        
        console.log(`🎯 注册下载节点: ${nodeData.name}`);
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            // 添加下载按钮
            this.addWidget("button", "📥 下载压缩包", "download_archive", () => {
                console.log("🐛 下载按钮被点击，检查节点状态...");
                console.log("🐛 节点images:", this.images);
                
                // 检查是否有images输出
                if (this.images && this.images.length > 0) {
                    console.log("✅ 检测到节点已执行，尝试下载...");
                    
                    if (this.images[0] && this.images[0].filename) {
                        const imageInfo = this.images[0];
                        downloadFile(
                            imageInfo.filename, 
                            imageInfo.type || "output", 
                            imageInfo.subfolder || ""
                        );
                    } else {
                        console.log("❌ 无法获取下载文件信息");
                        alert("无法获取下载文件信息，请重新执行节点！");
                    }
                } else {
                    console.log("❌ 节点未执行或无输出");
                    alert("请先执行节点生成下载文件！");
                }
            });
            
            return r;
        };
    }
});

console.log("✅ BatchVideoProcessor 下载处理扩展注册完成!");