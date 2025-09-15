"""
批量视频处理节点实现
"""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from typing_extensions import override

import folder_paths
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, io, ui

from .utils import (
    get_video_duration, get_video_info, scan_video_files,
    create_batch_folder, create_output_folder, prepare_end_video,
    cut_single_segment_with_end, create_download_archive,
    clean_old_batches, format_file_size
)


class BatchVideoUploader(io.ComfyNode):
    """批量视频上传节点"""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BatchVideoUploader",
            display_name="批量视频上传",
            category="batch_video",
            description="批量上传视频文件，支持文件夹选择和拖拽上传",
            inputs=[
                io.String.Input(
                    "session_name", 
                    default="batch_upload", 
                    tooltip="上传会话名称，用于组织文件"
                ),
                io.Combo.Input(
                    "upload_method", 
                    options=["folder_select", "file_list"], 
                    default="folder_select",
                    tooltip="上传方式：选择文件夹或文件列表"
                ),
                io.String.Input(
                    "source_folder",
                    default="",
                    tooltip="源文件夹路径（当选择folder_select时使用）"
                ),
                io.Boolean.Input(
                    "create_subfolder", 
                    default=True, 
                    tooltip="为每次上传创建子文件夹"
                ),
                io.String.Input(
                    "allowed_extensions", 
                    default="mp4,avi,mov,mkv,flv,wmv", 
                    tooltip="允许的视频格式（逗号分隔）"
                ),
                io.Int.Input(
                    "max_file_size_mb", 
                    default=500, 
                    min=1, 
                    max=2048, 
                    tooltip="单文件大小限制(MB)"
                ),
            ],
            outputs=[
                io.String.Output("upload_folder", display_name="上传文件夹路径"),
                io.Int.Output("uploaded_count", display_name="上传文件数"),
                io.String.Output("file_list", display_name="文件列表信息"),
            ],
        )

    @classmethod
    def execute(cls, session_name: str, upload_method: str, source_folder: str,
                create_subfolder: bool, allowed_extensions: str, max_file_size_mb: int) -> io.NodeOutput:
        
        input_dir = folder_paths.get_input_directory()
        
        # 创建批量上传目录
        if create_subfolder:
            upload_folder = create_batch_folder(input_dir, session_name)
        else:
            upload_folder = os.path.join(input_dir, "batch_uploads")
            os.makedirs(upload_folder, exist_ok=True)
        
        # 解析允许的扩展名
        extensions = [ext.strip() for ext in allowed_extensions.split(",")]
        max_size_bytes = max_file_size_mb * 1024 * 1024
        
        uploaded_files = []
        file_info_list = []
        
        if upload_method == "folder_select" and source_folder and os.path.exists(source_folder):
            # 扫描源文件夹中的视频文件
            video_files = scan_video_files(source_folder, extensions)
            
            for video_file in video_files:
                file_size = os.path.getsize(video_file)
                if file_size > max_size_bytes:
                    print(f"跳过文件 {video_file}: 大小超限 ({format_file_size(file_size)})")
                    continue
                
                # 复制文件到上传目录
                filename = os.path.basename(video_file)
                dest_path = os.path.join(upload_folder, filename)
                
                try:
                    import shutil
                    shutil.copy2(video_file, dest_path)
                    uploaded_files.append(dest_path)
                    
                    # 获取视频信息
                    video_info = get_video_info(dest_path)
                    file_info = {
                        "filename": filename,
                        "size": format_file_size(file_size),
                        "duration": f"{video_info.get('duration', 0):.2f}s",
                        "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
                        "path": dest_path
                    }
                    file_info_list.append(file_info)
                    
                except Exception as e:
                    print(f"复制文件失败 {video_file}: {e}")
        
        # 生成文件列表信息
        file_list_text = f"上传完成！\\n文件夹: {upload_folder}\\n\\n文件列表:\\n"
        for i, info in enumerate(file_info_list, 1):
            file_list_text += f"{i}. {info['filename']} - {info['size']} - {info['duration']} - {info['resolution']}\\n"
        
        return io.NodeOutput(
            upload_folder,
            len(uploaded_files),
            file_list_text
        )


class BatchVideoCutter(io.ComfyNode):
    """批量视频切分节点"""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BatchVideoCutter",
            display_name="批量视频切分",
            category="batch_video",
            description="批量切分视频并添加结尾视频",
            inputs=[
                io.String.Input(
                    "input_folder", 
                    tooltip="输入视频文件夹路径"
                ),
                io.Float.Input(
                    "cut_duration", 
                    default=30.0, 
                    min=1.0, 
                    max=300.0,
                    step=0.5,
                    tooltip="每段视频时长(秒)"
                ),
                io.Video.Input(
                    "end_video", 
                    tooltip="结尾视频"
                ),
                io.String.Input(
                    "output_prefix", 
                    default="processed_videos", 
                    tooltip="输出文件前缀"
                ),
                io.Int.Input(
                    "max_workers", 
                    default=4, 
                    min=1, 
                    max=16, 
                    tooltip="并行处理线程数"
                ),
                io.Boolean.Input(
                    "skip_short_videos", 
                    default=True, 
                    tooltip="跳过时长不足的视频"
                ),
                io.Float.Input(
                    "min_segment_ratio", 
                    default=0.5, 
                    min=0.1, 
                    max=1.0, 
                    step=0.1,
                    tooltip="最小片段比例"
                ),
            ],
            outputs=[
                io.String.Output("output_folder", display_name="输出文件夹"),
                io.Int.Output("total_segments", display_name="总片段数"),
                io.String.Output("summary", display_name="处理摘要"),
            ],
        )

    @classmethod
    def execute(cls, input_folder: str, cut_duration: float, end_video: VideoInput,
                output_prefix: str, max_workers: int, skip_short_videos: bool,
                min_segment_ratio: float) -> io.NodeOutput:
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        output_folder = create_output_folder(output_dir, output_prefix)
        
        # 扫描输入文件夹中的视频文件
        if not os.path.exists(input_folder):
            return io.NodeOutput(output_folder, 0, f"错误：输入文件夹不存在 {input_folder}")
        
        video_files = scan_video_files(input_folder)
        if not video_files:
            return io.NodeOutput(output_folder, 0, f"在文件夹 {input_folder} 中未找到视频文件")
        
        # 获取结尾视频路径
        end_video_path = end_video.get_path() if hasattr(end_video, 'get_path') else str(end_video)
        
        print(f"开始批量处理 {len(video_files)} 个视频文件")
        print(f"输出目录: {output_folder}")
        print(f"切分时长: {cut_duration}秒")
        print(f"线程数: {max_workers}")
        
        total_segments = 0
        processed_videos = 0
        failed_videos = []
        
        # 使用线程池处理视频
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for video_file in video_files:
                future = executor.submit(
                    cls._process_single_video,
                    video_file, cut_duration, end_video_path, output_folder,
                    skip_short_videos, min_segment_ratio
                )
                futures.append((future, video_file))
            
            # 收集结果
            for future, video_file in futures:
                try:
                    segments_count = future.result()
                    if segments_count > 0:
                        total_segments += segments_count
                        processed_videos += 1
                        print(f"✓ 处理完成: {os.path.basename(video_file)} ({segments_count} 个片段)")
                    else:
                        failed_videos.append(os.path.basename(video_file))
                except Exception as e:
                    print(f"✗ 处理失败: {os.path.basename(video_file)} - {e}")
                    failed_videos.append(os.path.basename(video_file))
        
        # 生成处理摘要
        summary = f"""批量处理完成！
输出目录: {output_folder}
处理视频: {processed_videos}/{len(video_files)}
总片段数: {total_segments}
切分时长: {cut_duration}秒"""
        
        if failed_videos:
            summary += f"\\n失败文件: {', '.join(failed_videos)}"
        
        return io.NodeOutput(output_folder, total_segments, summary)
    
    @staticmethod
    def _process_single_video(video_path: str, cut_duration: float, end_video_path: str,
                             output_folder: str, skip_short_videos: bool, 
                             min_segment_ratio: float) -> int:
        """处理单个视频文件"""
        video_name = Path(video_path).stem
        tid = threading.get_ident()
        video_duration = get_video_duration(video_path)
        
        if video_duration < cut_duration and skip_short_videos:
            print(f"[TID {tid}] 跳过视频 {video_name}: 时长 {video_duration:.2f}s 小于切分时长 {cut_duration}s")
            return 0
        
        # 计算可以切分的段数
        num_segments = int(video_duration // cut_duration)
        if num_segments * cut_duration >= video_duration:
            num_segments = max(1, num_segments - 1)
        
        if num_segments == 0:
            return 0
        
        print(f"[TID {tid}] 处理视频 {video_name}: 总时长 {video_duration:.2f}s, 将切分为 {num_segments} 段")
        
        # 创建视频输出目录
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        try:
            # 获取主视频信息用于预处理结尾视频
            video_info = get_video_info(video_path)
            main_width = video_info.get('width', 1920)
            main_height = video_info.get('height', 1080)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 预处理结尾视频
                prepared_end_path = prepare_end_video(end_video_path, main_width, main_height, temp_dir)
                end_duration = get_video_duration(prepared_end_path)
                
                segments_created = 0
                
                # 切分视频并添加结尾
                for i in range(num_segments):
                    start_time = i * cut_duration
                    end_time = (i + 1) * cut_duration
                    
                    # 确保切分时间不超过视频时长
                    if end_time > video_duration:
                        end_time = video_duration
                    
                    # 检查切分时长
                    if end_time - start_time < cut_duration * min_segment_ratio:
                        print(f"[TID {tid}] 跳过: segment_{i+1:03d}.mp4 (切分时长太短: {end_time - start_time:.2f}s)")
                        continue
                    
                    output_filename = f"segment_{i+1:03d}.mp4"
                    output_path = os.path.join(video_output_dir, output_filename)
                    
                    success = cut_single_segment_with_end(
                        video_path, start_time, end_time, output_path, 
                        prepared_end_path, end_duration
                    )
                    
                    if success:
                        segments_created += 1
                        print(f"[TID {tid}] 完成: {output_filename}")
                    else:
                        print(f"[TID {tid}] 失败: {output_filename}")
                
                return segments_created
                
        except Exception as e:
            print(f"[TID {tid}] 处理视频失败 {video_name}: {e}")
            return 0


class BatchVideoDownloader(io.ComfyNode):
    """批量视频下载节点"""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BatchVideoDownloader",
            display_name="批量视频下载",
            category="batch_video",
            description="批量打包下载处理后的视频文件",
            inputs=[
                io.String.Input(
                    "source_folder", 
                    tooltip="源文件夹路径（output目录下）"
                ),
                io.Combo.Input(
                    "download_format", 
                    options=["zip", "tar"], 
                    default="zip",
                    tooltip="压缩格式"
                ),
                io.String.Input(
                    "archive_name", 
                    default="processed_videos", 
                    tooltip="压缩包名称"
                ),
                io.Boolean.Input(
                    "include_metadata", 
                    default=True, 
                    tooltip="包含处理信息"
                ),
                io.Combo.Input(
                    "compression_level", 
                    options=["fast", "balanced", "best"], 
                    default="balanced",
                    tooltip="压缩级别"
                ),
            ],
            outputs=[
                io.String.Output("download_path", display_name="下载文件路径"),
                io.Int.Output("file_count", display_name="文件数量"),
                io.String.Output("archive_info", display_name="压缩包信息"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source_folder: str, download_format: str, archive_name: str,
                include_metadata: bool, compression_level: str) -> io.NodeOutput:
        
        if not os.path.exists(source_folder):
            return io.NodeOutput("", 0, f"错误：源文件夹不存在 {source_folder}")
        
        # 创建压缩包
        archive_path, file_count, total_size = create_download_archive(
            source_folder, archive_name, download_format, include_metadata
        )
        
        if not archive_path:
            return io.NodeOutput("", 0, "创建压缩包失败")
        
        # 生成压缩包信息
        archive_size = os.path.getsize(archive_path)
        archive_info = f"""下载包创建完成！
文件路径: {archive_path}
压缩格式: {download_format.upper()}
包含文件: {file_count} 个
原始大小: {format_file_size(total_size)}
压缩后大小: {format_file_size(archive_size)}
压缩率: {(1 - archive_size / total_size) * 100:.1f}%"""
        
        if include_metadata:
            archive_info += "\\n包含处理元数据: 是"
        
        return io.NodeOutput(
            archive_path,
            file_count,
            archive_info
        )


class BatchFileManager(io.ComfyNode):
    """批量文件管理节点"""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BatchFileManager",
            display_name="批量文件管理",
            category="batch_video",
            description="管理批量处理的文件和目录",
            inputs=[
                io.Combo.Input(
                    "action", 
                    options=["list_batches", "clean_old", "get_info"], 
                    default="list_batches",
                    tooltip="管理操作"
                ),
                io.Int.Input(
                    "days_to_keep", 
                    default=7, 
                    min=1, 
                    max=30,
                    tooltip="保留天数（用于清理操作）"
                ),
                io.String.Input(
                    "target_folder",
                    default="",
                    tooltip="目标文件夹（用于获取信息）"
                ),
            ],
            outputs=[
                io.String.Output("result", display_name="操作结果"),
                io.String.Output("details", display_name="详细信息"),
            ],
        )

    @classmethod
    def execute(cls, action: str, days_to_keep: int, target_folder: str) -> io.NodeOutput:
        
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        if action == "list_batches":
            # 列出所有批处理文件夹
            batch_folders = []
            
            # 检查输入目录中的批量上传
            batch_upload_dir = os.path.join(input_dir, "batch_uploads")
            if os.path.exists(batch_upload_dir):
                for item in os.listdir(batch_upload_dir):
                    item_path = os.path.join(batch_upload_dir, item)
                    if os.path.isdir(item_path):
                        file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                        batch_folders.append(f"📁 输入: {item} ({file_count} 文件)")
            
            # 检查输出目录中的处理结果
            batch_output_dir = os.path.join(output_dir, "processed_batches")
            if os.path.exists(batch_output_dir):
                for item in os.listdir(batch_output_dir):
                    item_path = os.path.join(batch_output_dir, item)
                    if os.path.isdir(item_path):
                        # 计算子文件夹和文件数量
                        total_files = sum([len([f for f in os.listdir(os.path.join(item_path, subdir)) 
                                              if os.path.isfile(os.path.join(item_path, subdir, f))])
                                          for subdir in os.listdir(item_path) 
                                          if os.path.isdir(os.path.join(item_path, subdir))])
                        batch_folders.append(f"📁 输出: {item} ({total_files} 文件)")
            
            result = f"找到 {len(batch_folders)} 个批处理文件夹"
            details = "\\n".join(batch_folders) if batch_folders else "无批处理文件夹"
            
        elif action == "clean_old":
            # 清理旧文件
            cleaned_folders = clean_old_batches(input_dir, days_to_keep)
            cleaned_folders.extend(clean_old_batches(output_dir, days_to_keep))
            
            result = f"清理完成，删除了 {len(cleaned_folders)} 个文件夹"
            details = "\\n".join([f"已删除: {folder}" for folder in cleaned_folders]) if cleaned_folders else "无需清理的文件夹"
            
        elif action == "get_info":
            # 获取文件夹信息
            if not target_folder or not os.path.exists(target_folder):
                result = "错误：目标文件夹不存在或未指定"
                details = f"指定的路径: {target_folder}"
            else:
                file_count = 0
                total_size = 0
                
                for root, dirs, files in os.walk(target_folder):
                    file_count += len(files)
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            pass
                
                result = f"文件夹信息获取完成"
                details = f"""路径: {target_folder}
文件数量: {file_count}
总大小: {format_file_size(total_size)}
子目录数: {len([d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))])}"""
        
        else:
            result = f"未知操作: {action}"
            details = "支持的操作: list_batches, clean_old, get_info"
        
        return io.NodeOutput(result, details)


class BatchVideoExtension(ComfyExtension):
    """批量视频处理扩展"""
    
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BatchVideoUploader,
            BatchVideoCutter,
            BatchVideoDownloader,
            BatchFileManager,
        ]