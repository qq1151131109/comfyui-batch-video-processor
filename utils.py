"""
批量视频处理工具函数
"""

import os
import glob
import tempfile
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import ffmpeg
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import folder_paths


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        print(f"获取视频时长失败 {video_path}: {e}")
        return 0


def get_video_info(video_path: str) -> dict:
    """获取视频详细信息"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        return {
            'duration': float(video_stream.get('duration', 0)),
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream.get('r_frame_rate', '30/1')),
            'codec': video_stream.get('codec_name', 'unknown'),
            'file_size': os.path.getsize(video_path)
        }
    except Exception as e:
        print(f"获取视频信息失败 {video_path}: {e}")
        return {}


def scan_video_files(folder_path: str, extensions: List[str] = None, recursive: bool = True) -> List[str]:
    """扫描文件夹中的视频文件
    
    Args:
        folder_path: 要扫描的文件夹路径
        extensions: 视频文件扩展名列表
        recursive: 是否递归扫描子目录，默认True
    """
    if extensions is None:
        extensions = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v']
    
    video_files = []
    
    if recursive:
        # 递归扫描所有子目录
        for ext in extensions:
            # 递归模式：使用 ** 通配符
            pattern = os.path.join(folder_path, "**", f"*.{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))
            # 同时支持大写扩展名
            pattern = os.path.join(folder_path, "**", f"*.{ext.upper()}")
            video_files.extend(glob.glob(pattern, recursive=True))
    else:
        # 非递归模式：只扫描当前目录
        for ext in extensions:
            pattern = os.path.join(folder_path, f"*.{ext}")
            video_files.extend(glob.glob(pattern))
            # 同时支持大写扩展名
            pattern = os.path.join(folder_path, f"*.{ext.upper()}")
            video_files.extend(glob.glob(pattern))
    
    return sorted(video_files)


def scan_media_files(folder_path: str, file_types: List[str] = None, recursive: bool = True) -> dict:
    """扫描文件夹中的多媒体文件
    
    Args:
        folder_path: 要扫描的文件夹路径
        file_types: 要扫描的文件类型列表 ['video', 'audio', 'image']
        recursive: 是否递归扫描子目录，默认True
    """
    if file_types is None:
        file_types = ['video', 'audio', 'image']
    
    # 定义各种文件类型的扩展名
    extensions = {
        'video': ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v', 'webm'],
        'audio': ['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a', 'wma'],
        'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    }
    
    result = {}
    total_files = []
    
    for file_type in file_types:
        if file_type not in extensions:
            continue
            
        files = []
        for ext in extensions[file_type]:
            if recursive:
                # 递归模式：使用 ** 通配符
                pattern = os.path.join(folder_path, "**", f"*.{ext}")
                files.extend(glob.glob(pattern, recursive=True))
                # 同时支持大写扩展名
                pattern = os.path.join(folder_path, "**", f"*.{ext.upper()}")
                files.extend(glob.glob(pattern, recursive=True))
            else:
                # 非递归模式：只扫描当前目录
                pattern = os.path.join(folder_path, f"*.{ext}")
                files.extend(glob.glob(pattern))
                # 同时支持大写扩展名
                pattern = os.path.join(folder_path, f"*.{ext.upper()}")
                files.extend(glob.glob(pattern))
        
        result[file_type] = sorted(files)
        total_files.extend(files)
    
    result['all'] = sorted(total_files)
    return result


def create_batch_folder(base_dir: str, session_name: str) -> str:
    """创建批量处理文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{session_name}"
    batch_folder = os.path.join(base_dir, "batch_uploads", folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    return batch_folder


def create_output_folder(base_dir: str, session_name: str) -> str:
    """创建输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{session_name}_processed"
    output_folder = os.path.join(base_dir, "processed_batches", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def prepare_end_video(end_video_path: str, target_width: int, target_height: int, temp_dir: str) -> str:
    """预处理结尾视频，统一编码参数"""
    try:
        prepared_path = os.path.join(temp_dir, "prepared_end.mp4")
        
        end_input_stream = ffmpeg.input(end_video_path)
        end_video_stream = (end_input_stream.video
                           .filter('scale', target_width, target_height, flags='lanczos')
                           .filter('setsar', '1'))
        end_audio_stream = end_input_stream.audio
        
        (
            ffmpeg
            .output(
                end_video_stream,
                end_audio_stream,
                prepared_path,
                vcodec='libx264',
                preset='fast',
                **{'profile:v': 'main'},
                r=30,
                acodec='aac',
                ar=44100,
                ac=2
            )
            .overwrite_output()
            .run(quiet=True)
        )
        
        return prepared_path
    except Exception as e:
        print(f"预处理结尾视频失败: {e}")
        return end_video_path


def cut_single_segment_with_end(video_path: str, start_time: float, end_time: float, 
                               output_path: str, prepared_end_path: str, end_duration: float) -> bool:
    """切分单个视频片段并添加结尾视频"""
    try:
        # 获取主视频信息
        main_probe = ffmpeg.probe(video_path)
        main_video_stream = next(s for s in main_probe['streams'] if s['codec_type'] == 'video')
        main_width = int(main_video_stream['width'])
        main_height = int(main_video_stream['height'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_main = os.path.join(temp_dir, "temp_main.mp4")
            
            # 切分主视频
            input_stream = ffmpeg.input(video_path, ss=start_time, t=end_time-start_time, 
                                       **{'fflags': '+ignidx+igndts'})
            video_stream = (input_stream.video
                           .filter('scale', main_width, main_height, flags='lanczos')
                           .filter('setsar', '1'))
            audio_stream = input_stream.audio
            
            (
                ffmpeg
                .output(
                    video_stream,
                    audio_stream,
                    temp_main,
                    vcodec='libx264',
                    preset='fast',
                    **{'profile:v': 'main'},
                    r=30,
                    acodec='aac',
                    ar=44100,
                    ac=2,
                    **{'fflags': '+ignidx+igndts'}
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 合并主视频和结尾视频
            main_input = ffmpeg.input(temp_main)
            end_input = ffmpeg.input(prepared_end_path)
            
            (
                ffmpeg
                .filter([main_input.video, main_input.audio, end_input.video, end_input.audio], 
                       'concat', n=2, v=1, a=1)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
        
        return True
    except Exception as e:
        print(f"处理视频片段失败 {video_path}: {e}")
        return False


def create_download_archive(source_folder: str, archive_name: str, archive_format: str = "zip", 
                          include_metadata: bool = True) -> Tuple[str, int, int]:
    """创建下载压缩包"""
    try:
        output_dir = folder_paths.get_output_directory()
        # 直接使用output目录，避免子目录问题
        print(f"🐛 调试: output_dir = {output_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        # 使用英文文件名，避免编码问题
        safe_archive_name = "batch_result" if any(ord(c) > 127 for c in archive_name) else archive_name
        archive_path = os.path.join(output_dir, f"{safe_archive_name}_{timestamp}.{archive_format}")
        print(f"🐛 调试: archive_path = {archive_path}")
        
        file_count = 0
        total_size = 0
        
        if archive_format == "zip":
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(source_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_folder)
                        zipf.write(file_path, arcname)
                        file_count += 1
                        total_size += os.path.getsize(file_path)
                
                # 添加元数据文件
                if include_metadata:
                    metadata = generate_processing_metadata(source_folder)
                    zipf.writestr("processing_info.json", metadata)
        
        return archive_path, file_count, total_size
    except Exception as e:
        print(f"创建压缩包失败: {e}")
        return "", 0, 0


def generate_processing_metadata(folder_path: str) -> str:
    """生成处理元数据"""
    import json
    
    metadata = {
        "processing_time": datetime.now().isoformat(),
        "folder_path": folder_path,
        "file_count": len(os.listdir(folder_path)),
        "total_size": sum(os.path.getsize(os.path.join(folder_path, f)) 
                         for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))),
        "version": "1.0"
    }
    
    return json.dumps(metadata, indent=2, ensure_ascii=False)


def clean_old_batches(base_dir: str, days_to_keep: int = 7) -> List[str]:
    """清理旧的批处理文件"""
    import time
    
    cleaned_folders = []
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    for folder_type in ["batch_uploads", "processed_batches"]:
        folder_path = os.path.join(base_dir, folder_type)
        if not os.path.exists(folder_path):
            continue
            
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                if os.path.getctime(item_path) < cutoff_time:
                    try:
                        shutil.rmtree(item_path)
                        cleaned_folders.append(item_path)
                    except Exception as e:
                        print(f"清理文件夹失败 {item_path}: {e}")
    
    return cleaned_folders


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"