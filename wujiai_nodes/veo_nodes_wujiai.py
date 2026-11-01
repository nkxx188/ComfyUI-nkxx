# wujiai_nodes/veo_nodes_wujiai.py
import os
import requests
import time
import uuid
import folder_paths
import traceback
from comfy.comfy_types import IO
from datetime import datetime

# 从公共工具箱导入核心底座
from ..utils import (
    TaskManager,
    upload_image_wujiai,
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
veo_task_db = TaskManager("wujiai_veo_history.json", max_completed_history=15)

def _get_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

# ======================================================================
# 节点 1: Veo 提交任务
# ======================================================================
class VeoSubmitTask:
    CATEGORY = "Nkxx/Wujiai/视频"
    
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [
            "veo_3_1-fast", 
            "veo_3_1",  
            "veo_3_1-fast-4K", 
            "veo_3_1-4K",
            "veo_3_1-components",
            "veo_3_1-components-4K",
            "veo_3_1-fast-components-4K",
            "veo_3_1-lite",          # 新增
            "veo_3_1-lite-4K"        # 新增
        ]
        return { 
            "required": {
                "prompt": ("STRING", {"default": "A cinematic shot of...", "multiline": True}),
                "model": (model_list, {"default": "veo_3_1-fast"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            }, 
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取全局配置"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()
    
    def submit(self, prompt, model, aspect_ratio, api_key="", image_1=None, image_2=None, image_3=None):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return ("请配置 Wujiai API Key",)
        
        try:
            print(f"[Wujiai Veo] 提交任务，模型: {model}")
            
            payload = {
                "model": model,
                "prompt": prompt.strip(),
                "aspect_ratio": aspect_ratio,
                "enable_upsample": True,
                "enhance_prompt": True
            }
            
            uploaded_images = []
            def process_img(img_input):
                if img_input is not None:
                    return upload_image_wujiai(final_api_key, img_input)
                return None

            url1 = process_img(image_1)
            if url1: uploaded_images.append(url1)
            url2 = process_img(image_2)
            if url2: uploaded_images.append(url2)
            url3 = process_img(image_3)
            if url3: uploaded_images.append(url3)

            if uploaded_images:
                payload["images"] = uploaded_images

            url = f"{HOST}/v1/video/create"
            res = requests.post(url, headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            
            data = res.json()
            task_id = data.get("id") or data.get("data", {}).get("id")
            
            if not task_id:
                raise Exception(f"API未返回任务ID: {data}")
            
            veo_task_db.update_task(str(task_id), {
                "prompt": prompt,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "model": model
            })
                
            return (f"Veo 任务提交成功!\n模型: {model}\nID: {task_id}",)
            
        except Exception as e:
            traceback.print_exc()
            return (f"提交失败: {e}",)

# ======================================================================
# 节点 2: Veo 查询状态
# ======================================================================
class VeoQueryTasks:
    CATEGORY = "Nkxx/Wujiai/视频"

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "query"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def query(self, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return ("请配置 Wujiai API Key",)

        with veo_task_db.lock:
            tasks = veo_task_db.read_tasks()
        
        pending_tasks = {k: v for k, v in tasks.items() if v.get("status") in ["running", "pending", "processing"]}
        
        for tid, tinfo in pending_tasks.items():
            try:
                query_url = f"{HOST}/v1/video/query?id={tid}"
                res = requests.get(query_url, headers=_get_headers(final_api_key), timeout=30)
                
                if res.status_code == 200:
                    d = res.json()
                    
                    raw_status = d.get("status") or d.get("data", {}).get("status") or "running"
                    raw_status = str(raw_status).lower()
                    
                    video_url = d.get("data", {}).get("video_url") or d.get("video_url")
                    
                    is_success = raw_status in ["success", "succeeded", "completed"]
                    is_failed = any(x in raw_status for x in ["failed", "failure", "error"])
                    
                    new_internal_status = "running"
                    updates = {}
                    
                    if is_success and video_url:
                        new_internal_status = "succeeded"
                        updates["video_url"] = video_url
                    elif is_failed:
                        new_internal_status = "failed"
                    
                    if tasks[tid]["status"] != new_internal_status:
                        updates["status"] = new_internal_status
                        veo_task_db.update_task(tid, updates)
                        
            except Exception as e:
                print(f"[Wujiai Veo] 查询 {tid} 失败: {e}")
        
        final_tasks = veo_task_db.read_tasks()
        report = ["--- Veo 任务队列 (Wujiai) ---"]
        
        for tid, info in sorted(final_tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
            status = info.get("status", "unknown")
            prompt_short = info.get("prompt", "")[:20].replace("\n", "")
            model_info = info.get("model", "")
            
            display_status = "Running"
            if status == "succeeded": display_status = "Succeeded"
            elif status == "failed": display_status = "Failed"
            elif status == "downloaded": display_status = "Downloaded"
            elif status == "download_failed": display_status = "Download Failed"
            
            report.append(f"[{display_status}] ({model_info}) : {prompt_short}...")
            
        return ("\n".join(report),)

# ======================================================================
# 节点 3: Veo 下载视频
# ======================================================================
class VeoGetNextVideo:
    CATEGORY = "Nkxx/Wujiai/视频"

    @classmethod
    def INPUT_TYPES(cls): return {}
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "get_video"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_video(self):
        with veo_task_db.lock:
            tasks = veo_task_db.read_tasks()
        
        target_id = None
        for tid, info in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', '0')):
            if info.get("status") == "succeeded":
                target_id = tid
                break
        
        if not target_id:
            return (VideoAdapter(""), "没有等待下载的新完成任务。")
            
        url = tasks[target_id].get("video_url")
        output_dir = folder_paths.get_output_directory()
        filename = f"veo_{target_id[:8]}_{uuid.uuid4().hex[:4]}.mp4"
        path = os.path.join(output_dir, filename)
        
        try:
            robust_download_video(url, path)
            veo_task_db.update_task(target_id, {"status": "downloaded", "video_path": path})
            return (VideoAdapter(path), f"下载成功: {filename}")
        except Exception as e:
            veo_task_db.update_task(target_id, {"status": "download_failed"})
            return (VideoAdapter(""), f"自动下载失败: {e}\n\n您可以复制以下链接手动下载:\n{url}")

# --- 注册 ---
NODE_CLASS_MAPPINGS = {
    "VeoSubmitTask": VeoSubmitTask,
    "VeoQueryTasks": VeoQueryTasks,
    "VeoGetNextVideo": VeoGetNextVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoSubmitTask": "1. Veo 提交任务 (Wujiai)",
    "VeoQueryTasks": "2. Veo 查询状态 (Wujiai)",
    "VeoGetNextVideo": "3. Veo 下载视频 (Wujiai)",
}
