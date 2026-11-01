# ComfyUI-nkxx/utils.py
import os
import requests
import time
import json
import threading
import tempfile
import shutil
import subprocess
import io
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import traceback
import yt_dlp
import ssl
import urllib3

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

def resolve_api_key(api_key_from_node: str, channel: str = "grsai") -> str:
    """
    Resolve an API key through the global Nkxx hook while staying compatible
    with older get_api_key implementations that only accepted one argument.
    """
    import builtins

    getter = getattr(builtins, "get_api_key", None)
    if callable(getter):
        try:
            return getter(api_key_from_node, channel)
        except TypeError as exc:
            if "positional argument" not in str(exc) and "positional arguments" not in str(exc):
                raise
            return getter(api_key_from_node)

    if api_key_from_node and api_key_from_node.strip():
        return api_key_from_node.strip()

    if channel.lower() == "grsai":
        return os.getenv("GRSAI_KEY", "").strip()
    if channel.lower() == "wujiai":
        return os.getenv("WUJIAI_KEY", "").strip()
    return ""

# 零宽字符，用于防止 API 缓存
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]
GRSAI_LLM_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro",
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gemini-3-flash",
    "gemini-3-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
# ======================================================================
# 动态防封路由配置 (全局单例)
# ======================================================================
GITEE_URL = "https://gitee.com/jiege777/app-config/raw/master/api_url.txt"
_W_HOST_CACHE = None  # 全局内存缓存

def get_wujiai_host() -> str:
    """动态获取最新的 API 域名，支持防封热切换"""
    global _W_HOST_CACHE
    if _W_HOST_CACHE:
        return _W_HOST_CACHE
    
    fallback = "https://wujiai.org" 
    try:
        res = requests.get(GITEE_URL, timeout=3)
        if res.status_code == 200:
            url = res.text.strip().rstrip('/')
            if url.startswith("http"):
                _W_HOST_CACHE = url
                print(f"[Wujiai Core] 🌍 动态路由获取成功: {url}")
                return url
    except Exception:
        pass
    
    _W_HOST_CACHE = fallback
    print(f"[Wujiai Core] ⚠️ 动态路由获取失败，使用备用通道: {fallback}")
    return fallback
# ======================================================================
# 图像与张量处理 (Image & Tensor Utils)
# ======================================================================
def download_image(url: str, timeout: int = 60) -> Image.Image | None:
    """从URL下载图像并返回PIL.Image对象"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"[Image Utils] 图像下载失败: {e}")
        return None

def safe_pil_to_rgb(image: Image.Image) -> Image.Image:
    """安全地将任何PIL图像转换为RGB模式"""
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image

def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """将torch张量 (B, H, W, C) 转换为PIL图像列表"""
    if not isinstance(tensor, torch.Tensor): return []
    images = []
    for i in range(tensor.shape[0]):
        img_np = (torch.clamp(tensor[i], 0, 1).cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA'))
    return images

def pil_to_tensor(pil_images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """将PIL图像或列表转换为ComfyUI图像张量"""
    if not isinstance(pil_images, list): pil_images = [pil_images]
    tensors = []
    for pil_image in pil_images:
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)[None,]
        tensors.append(tensor)
    if not tensors: return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    return torch.cat(tensors, dim=0)

def safe_pil_batch_to_tensor(pil_images: list[Image.Image]) -> torch.Tensor:
    """将PIL图像列表安全地转换为ComfyUI图像张量，自动处理尺寸不一致（填充黑边）"""
    if not pil_images:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    tensors, max_h, max_w = [], 0, 0
    for pil_image in pil_images:
        if pil_image is None: continue
        try:
            pil_image = safe_pil_to_rgb(pil_image)
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            
            if len(img_array.shape) == 2: 
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[2] == 4: 
                img_array = img_array[:,:,:3]

            tensor = torch.from_numpy(img_array)[None,]
            max_h = max(max_h, tensor.shape[1])
            max_w = max(max_w, tensor.shape[2])
            tensors.append(tensor)
        except Exception as e:
            print(f"[Image Utils] 跳过损坏的图像: {e}")

    if not tensors: return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    
    padded_tensors = []
    for tensor in tensors:
        _, h, w, c = tensor.shape
        if h == max_h and w == max_w:
            padded_tensors.append(tensor)
            continue
        tensor_chw = tensor.permute(0, 3, 1, 2)
        pad_w, pad_h = max_w - w, max_h - h
        padded_tensor_chw = F.pad(tensor_chw, (0, pad_w, 0, pad_h), "constant", 0)
        padded_tensors.append(padded_tensor_chw.permute(0, 2, 3, 1))

    return torch.cat(padded_tensors, dim=0)

# ======================================================================
# 双渠道图片上传 (Uploaders)
# ======================================================================
def upload_image_grsai(api_key: str, image_input: torch.Tensor | Image.Image) -> str | None:
    """Grsai 渠道的图片上传逻辑 (获取Token -> 上传文件)"""
    try:
        if isinstance(image_input, torch.Tensor):
            pil_image = tensor_to_pil(image_input)[0]
        else:
            pil_image = image_input
            
        rgb_pil = safe_pil_to_rgb(pil_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
            rgb_pil.save(temp_f, "PNG")
            temp_path = temp_f.name

        host = "https://grsai.dakka.com.cn"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # 1. 获取 Token
        token_res = requests.post(f"{host}/client/resource/newUploadTokenZH", headers=headers, json={"sux": "png"}, timeout=30)
        token_res.raise_for_status()
        token_data = token_res.json().get("data", {})
        
        # 2. 上传文件
        with open(temp_path, "rb") as f:
            up_res = requests.post(
                url=token_data["url"], 
                data={"token": token_data["token"], "key": token_data["key"]}, 
                files={"file": f}, 
                timeout=120
            )
            up_res.raise_for_status()
            
        os.unlink(temp_path)
        return f"{token_data['domain']}/{token_data['key']}"
    except Exception as e:
        print(f"[Grsai Upload] 上传失败: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None

def upload_image_wujiai(api_key: str, image_input: torch.Tensor | Image.Image) -> str | None:
    """Wujiai 渠道的图片上传逻辑 """
    print(f"[Wujiai Upload] 开始处理图片...")
    try:
        if isinstance(image_input, torch.Tensor):
            if len(image_input.shape) == 4:
                image_input = image_input[0]
            arr = np.clip(image_input.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
        else:
            image = image_input

        image = safe_pil_to_rgb(image)        
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        
        url = "https://imageproxy.zhongzhuan.chat/api/upload"
        headers = {'Authorization': f"Bearer {api_key}"}
        
        for attempt in range(3):
            try:
                buf.seek(0)
                response = requests.post(url, headers=headers, files={'file': ('image.png', buf, 'image/png')}, timeout=45)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("data") or data.get("url")
            except Exception:
                time.sleep(1)
        raise Exception("多次尝试均失败")
    except Exception as e:
        print(f"[Wujiai Upload] 流程出错: {e}")
        return None

# ======================================================================
# 视频下载器与适配器 (Video Downloader & Adapter)
# ======================================================================
def robust_download_video(video_url: str, output_path: str, max_retries: int = 3, timeout: int = 300) -> bool:
    """强鲁棒性视频下载器: 优先使用 yt-dlp，失败则回退使用 curl"""
    print(f"[Downloader] 准备下载: {video_url}")
    
    # 方案 1: yt-dlp
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        ydl_opts = {
            'outtmpl': output_path,
            'retries': max_retries,
            'http_headers_str': "\r\n".join([f"{k}: {v}" for k, v in headers.items()]),
            'quiet': True,
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'socket_timeout': 30,
            'nocheckcertificate': True, 
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            print("[Downloader] ✓ yt-dlp 下载成功")
            return True
    except Exception as e:
        print(f"[Downloader] yt-dlp 失败: {e}。切换至 Curl 备用方案。")
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except: pass

    # 方案 2: Curl
    curl_path = shutil.which("curl")
    if curl_path:
        for attempt in range(max_retries):
            try:
                print(f"[Downloader] 尝试调用系统 Curl (第 {attempt+1}/{max_retries} 次)...")
                cmd = [curl_path, "-k", "-L", "--connect-timeout", "20", "--max-time", str(timeout), "-o", output_path, video_url]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout + 10, check=True)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    print("[Downloader] ✓ Curl 下载成功")
                    return True
            except Exception:
                time.sleep(2)
                
    raise Exception("所有下载方案均失败 (yt-dlp 和 Curl)")

class VideoAdapter:
    """通用的视频对象包装器，供 ComfyUI 原生 Save Video 节点读取"""
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_dimensions(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path) or os.path.getsize(self.video_path) < 1024:
                return 1280, 720
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened(): return 1280, 720
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        except Exception:
            return 1280, 720 

    def save_to(self, output_path, **kwargs):
        if self.video_path and os.path.exists(self.video_path):
            shutil.copyfile(self.video_path, output_path)
            return True
        return False

# ======================================================================
# 统一的异步任务管理器 (Task Manager)
# ======================================================================
class TaskManager:
    """管理异步任务记录（JSON），内部已包含线程锁机制，防止文件读写冲突"""
    def __init__(self, filename: str, max_completed_history: int = 15):
        # 1. 获取当前 utils.py 所在的目录 (ComfyUI-nkxx)
        base_dir = os.path.dirname(__file__)
        
        # 2. 自动创建一个专属的 data 文件夹存放所有 json 任务文件
        self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 3. 将 JSON 文件路径指向 data 文件夹内部
        self.task_file = os.path.join(self.data_dir, filename)
        
        self.max_history = max_completed_history
        self.lock = threading.Lock()
        
        # 常见的所有“活跃中”的状态集合
        self.active_statuses = {"pending", "running", "processing", "SUBMITTED", "IN_PROGRESS"}

    def read_tasks(self) -> dict:
        """读取任务字典"""
        if not os.path.exists(self.task_file): 
            return {}
        try:
            with open(self.task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def write_tasks(self, tasks: dict):
        """写入任务字典（自动清理历史）"""
        try:
            trimmed_tasks = self._trim_history(tasks)
            with open(self.task_file, 'w', encoding='utf-8') as f:
                json.dump(trimmed_tasks, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Task Manager] 写入任务文件失败: {e}")

    def update_task(self, task_id: str, updates: dict):
        """安全地原子化更新单个任务信息"""
        with self.lock:
            tasks = self.read_tasks()
            if task_id not in tasks:
                tasks[task_id] = {}
            tasks[task_id].update(updates)
            self.write_tasks(tasks)

    def _trim_history(self, tasks: dict) -> dict:
        """内部方法：清理过多的已完成任务"""
        active_tasks = {tid: t for tid, t in tasks.items() if t.get("status") in self.active_statuses}
        completed_tasks = {tid: t for tid, t in tasks.items() if t.get("status") not in self.active_statuses}
        
        if len(completed_tasks) > self.max_history:
            # 按提交时间倒序排
            sorted_completed = sorted(completed_tasks.items(), key=lambda x: x[1].get('submitted_at', '0'), reverse=True)
            kept_completed = dict(sorted_completed[:self.max_history])
            return {**active_tasks, **kept_completed}
        return tasks
