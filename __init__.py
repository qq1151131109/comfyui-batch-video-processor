"""
批量视频处理扩展
支持批量上传、切分处理、批量下载视频文件
"""

from .nodes import (
    BatchVideoLoader, BatchVideoComposer, BatchVideoCutter, 
    BatchVideoCropper, VideoNormalizer, TraverseVideoConcatenator,
    FileListConcatenator, BatchVideoDownloader, FileListDownloader, VideoStaticCleaner, 
    GameHighlightExtractor, VideoThumbnailGenerator, BatchLLMGenerator, 
    BatchTTSGenerator, SmartVideoCutterWithAudio
)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "BatchVideoLoader": BatchVideoLoader,
    "BatchVideoComposer": BatchVideoComposer,
    "BatchVideoCutter": BatchVideoCutter,
    "BatchVideoCropper": BatchVideoCropper,
    "VideoNormalizer": VideoNormalizer,
    "TraverseVideoConcatenator": TraverseVideoConcatenator,
    "FileListConcatenator": FileListConcatenator,
    "BatchVideoDownloader": BatchVideoDownloader,
    "FileListDownloader": FileListDownloader,
    "VideoStaticCleaner": VideoStaticCleaner,
    "GameHighlightExtractor": GameHighlightExtractor,
    "VideoThumbnailGenerator": VideoThumbnailGenerator,
    "BatchLLMGenerator": BatchLLMGenerator,
    "BatchTTSGenerator": BatchTTSGenerator,
    "SmartVideoCutterWithAudio": SmartVideoCutterWithAudio,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchVideoLoader": "🎬 批量素材加载器",
    "BatchVideoComposer": "🖼️ 视频画面拼接器", 
    "BatchVideoCutter": "✂️ 批量视频切分器",
    "BatchVideoCropper": "🔲 批量视频裁剪器",
    "VideoNormalizer": "📱 TikTok格式转换器",
    "TraverseVideoConcatenator": "🎲 批量视频拼接器",
    "FileListConcatenator": "🎯 文件列表拼接器",
    "BatchVideoDownloader": "📥 批量视频下载器",
    "FileListDownloader": "🎯 文件列表下载器",
    "VideoStaticCleaner": "⚡ 卡顿修复器",
    "GameHighlightExtractor": "🏆 游戏高光提取器",
    "VideoThumbnailGenerator": "🖼️ 视频缩略图生成器",
    "BatchLLMGenerator": "🤖 批量LLM文案生成器",
    "BatchTTSGenerator": "🎤 批量TTS语音生成器", 
    "SmartVideoCutterWithAudio": "🎧 音频时长匹配器",
}

# 指定前端扩展目录
WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]