# wujiai_nodes/grok_nodes_wujiai.py
import os
import requests
import time
import uuid
import folder_paths
import traceback
import concurrent.futures 
import pandas as pd
from datetime import datetime
from comfy.comfy_types import IO

import base64
import io
import numpy as np
from PIL import Image

# 从公共工具箱导入核心底座 (移除了 upload_image_wujiai)
from ..utils import (
    TaskManager,
    robust_download_video,
    VideoAdapter,
    get_wujiai_host
)

# --- 全局配置 ---
def resolve_api_key(api_key_from_node: str, channel: str = "grsai") -> str:
    import builtins
    import os

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

HOST = get_wujiai_host()
# 统一管理 Grok 专属任务
grok_task_db = TaskManager("wujiai_grok_history.json", max_completed_history=15)

def _get_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

def tensor_to_base64(image_tensor):
    """将 ComfyUI 的图像 Tensor 直接转换为 Base64 字符串 (保持无损 PNG)"""
    # 消除 Batch 维度
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # 转换为 PIL Image
    i = 255. * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
    # 转为 Base64 (PNG格式，无压缩)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

# ======================================================================
# 节点 1: Grok 提交任务
# ======================================================================
class GrokSubmitTask:
    CATEGORY = "Nkxx/Wujiai/视频"
    
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "prompt": ("STRING", {"default": "A cute cat", "multiline": True}),
                "aspect_ratio": (["2:3", "3:2", "1:1", "9:16", "16:9"], {"default": "16:9"}),
                "duration": (["6", "10", "15"], {"default": "6"}),
            }, "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取全局配置"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()
    
    def submit(self, prompt, aspect_ratio, duration, api_key="", images=None):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return ("请配置 Wujiai API Key",)
        
        try:
            model_name = "grok-video-3"
            if str(duration) == "10": model_name = "grok-video-3-10s"
            elif str(duration) == "15": model_name = "grok-video-3-15s"
            
            print(f"[Wujiai Grok] 提交任务，时长: {duration}s, 模型: {model_name}")

            # 恢复最初始的参数结构
            payload = {
                "model": model_name,
                "prompt": prompt.strip(),
                "aspect_ratio": aspect_ratio,
                "size": "720P" 
            }
            
            if images is not None:
                # 兼容列表与 ComfyUI 原生 Batch
                items = images if isinstance(images, list) else [images]
                flat_images = []
                for item in items:
                    if hasattr(item, 'shape') and len(item.shape) == 4 and item.shape[0] > 1:
                        for i in range(item.shape[0]):
                            flat_images.append(item[i:i+1])
                    else:
                        flat_images.append(item)
                
                # Grok 限制最多 7 张
                flat_images = flat_images[:7]
                base64_images = []
                
                for idx, img in enumerate(flat_images):
                    try:
                        b64_data = tensor_to_base64(img)
                        base64_images.append(b64_data)
                        print(f"[Wujiai Grok] 图片 {idx+1} 已本地转换为 Base64 (PNG)")
                    except Exception as e:
                        raise Exception(f"第 {idx+1} 张图片 Base64 转换失败: {e}")
                
                if base64_images:
                    payload["images"] = base64_images

            url = f"{HOST}/v1/video/create"
            res = requests.post(url, headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            
            data = res.json()
            task_id = data.get("id") or data.get("data", {}).get("id")
            
            if not task_id:
                raise Exception(f"API未返回任务ID: {data}")
            
            grok_task_db.update_task(str(task_id), {
                "prompt": prompt,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running", 
                "model": model_name
            })
                
            return (f"Grok 任务提交成功!\n模型: {model_name}\nID: {task_id}",)
            
        except Exception as e:
            traceback.print_exc()
            return (f"提交失败: {e}",)

# ======================================================================
# 节点 1.5: Grok 批量提交
# ======================================================================
class GrokSubmitBatchTask:
    CATEGORY = "Nkxx/Wujiai/视频"
    
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "CSV/Excel文件路径"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "aspect_ratio": (["2:3", "3:2", "1:1", "9:16", "16:9"], {"default": "16:9"}),
                "duration": (["6", "10", "15"], {"default": "6"}),
                "concurrency": ("INT", {"default": 2, "min": 1, "max": 10}),
            }, "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_ids", "report")
    FUNCTION = "submit_batch"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit_batch(self, file_path, column_name, aspect_ratio, duration, concurrency, api_key="", images=None):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return ("", "未配置 Wujiai API Key")
        
        model_name = "grok-video-3"
        if str(duration) == "10": model_name = "grok-video-3-10s"
        elif str(duration) == "15": model_name = "grok-video-3-15s"
        
        print(f"[Wujiai Grok] 批量提交开始，目标模型: {model_name}")

        try:
            if file_path.lower().endswith('.csv'): df = pd.read_csv(file_path)
            else: df = pd.read_excel(file_path)
            prompts = df[column_name].dropna().astype(str).tolist()
        except Exception as e: 
            return ("", f"读取文件错误: {e}")

        base64_images = []
        if images is not None:
            try:
                items = images if isinstance(images, list) else [images]
                flat_images = []
                for item in items:
                    if hasattr(item, 'shape') and len(item.shape) == 4 and item.shape[0] > 1:
                        for i in range(item.shape[0]):
                            flat_images.append(item[i:i+1])
                    else:
                        flat_images.append(item)
                
                flat_images = flat_images[:7]
                
                for idx, img in enumerate(flat_images):
                    b64_data = tensor_to_base64(img)
                    base64_images.append(b64_data)
            except Exception as e:
                return ("", f"批量任务图片 Base64 转换失败: {e}")

        def _worker(p):
            try:
                payload = {
                    "model": model_name, 
                    "prompt": p, 
                    "aspect_ratio": aspect_ratio, 
                    "size": "720P"
                }
                if base64_images: payload["images"] = base64_images
                
                res = requests.post(f"{HOST}/v1/video/create", headers=_get_headers(final_api_key), json=payload, timeout=60)
                res.raise_for_status()
                d = res.json()
                tid = d.get("id") or d.get("data", {}).get("id")
                
                if not tid: return (p, "无ID返回")
                
                grok_task_db.update_task(str(tid), {
                    "prompt": p, 
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    "status": "running",
                    "model": model_name
                })
                return (p, tid)
            except Exception as e:
                return (p, f"Err: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as exc:
            results = list(exc.map(_worker, prompts))
            
        return ("\n".join([str(r) for r in results]), f"批量处理完成: {len(results)} 条")

# ======================================================================
# 节点 2: Grok 查询状态
# ======================================================================
class GrokQueryTasks:
    CATEGORY = "Nkxx/Wujiai/视频"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "query"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def query(self, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return ("请配置 Wujiai API Key",)

        with grok_task_db.lock:
            tasks = grok_task_db.read_tasks()
            
        pending_tasks = {k: v for k, v in tasks.items() if v.get("status") in ["pending", "running", "processing"]}
        
        for tid, tinfo in pending_tasks.items():
            try:
                query_url = f"{HOST}/v1/video/query?id={tid}"
                res = requests.get(query_url, headers=_get_headers(final_api_key), timeout=30)
                if res.status_code == 200:
                    d = res.json()
                    new_status = str(d.get("status", "running")).lower()
                    
                    if new_status == "pending": new_status = "running"

                    if new_status in ["success", "succeeded", "completed"] and d.get("video_url"):
                        grok_task_db.update_task(tid, {"status": "succeeded", "video_url": d.get("video_url")})
                    elif new_status in ["failed", "failure", "error"]:
                        grok_task_db.update_task(tid, {"status": "failed"})
                    elif tasks[tid]["status"] != new_status:
                        grok_task_db.update_task(tid, {"status": new_status})
                        
            except Exception as e:
                print(f"[Wujiai Grok] 查询 {tid} 失败: {e}")
        
        # 读取最新状态生成报告
        final_tasks = grok_task_db.read_tasks()
        report = ["--- Grok 任务队列 (Wujiai) ---"]
        
        for tid, info in sorted(final_tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
            status = info.get("status", "unknown").capitalize()
            prompt_short = info.get("prompt", "")[:20].replace("\n", "")
            model_info = info.get("model", "")
            
            report.append(f"[{status}] ({model_info}) : {prompt_short}...")
            
        return ("\n".join(report),)

# ======================================================================
# 节点 3: Grok 下载视频
# ======================================================================
class GrokGetNextVideo:
    CATEGORY = "Nkxx/Wujiai/视频"
    
    @classmethod
    def INPUT_TYPES(cls): return {}
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "get_video"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_video(self):
        with grok_task_db.lock:
            tasks = grok_task_db.read_tasks()
            
        target_id = next((k for k, v in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', '')) if v.get("status") == "succeeded"), None)
        
        if not target_id:
            return (VideoAdapter(""), "没有等待下载的新完成任务。")
            
        url = tasks[target_id].get("video_url")
        output_dir = folder_paths.get_output_directory()
        filename = f"grok_{target_id[:8]}_{uuid.uuid4().hex[:4]}.mp4"
        path = os.path.join(output_dir, filename)
        
        try:
            robust_download_video(url, path)
            grok_task_db.update_task(target_id, {"status": "downloaded", "video_path": path})
            return (VideoAdapter(path), f"下载成功: {filename}")
        except Exception as e:
            grok_task_db.update_task(target_id, {"status": "download_failed"})
            return (VideoAdapter(""), f"下载失败: {e}\nURL: {url}")

# --- 注册 ---
NODE_CLASS_MAPPINGS = {
    "GrokSubmitTask": GrokSubmitTask,
    "GrokSubmitBatchTask": GrokSubmitBatchTask,
    "GrokQueryTasks": GrokQueryTasks,
    "GrokGetNextVideo": GrokGetNextVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokSubmitTask": "1.Grok 提交任务 (Wujiai)",
    "GrokSubmitBatchTask": "1.5.Grok 批量提交 (Wujiai)",
    "GrokQueryTasks": "2.Grok 查询状态 (Wujiai)",
    "GrokGetNextVideo": "3.Grok 下载视频 (Wujiai)",
}
