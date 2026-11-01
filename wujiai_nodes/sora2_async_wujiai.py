import os
import requests
import time
import uuid
import folder_paths
import traceback
import secrets
from datetime import datetime
import pandas as pd
import concurrent.futures
from comfy.comfy_types import IO

try:
    from .utils import (
        get_wujiai_host,
        upload_image_wujiai,
        robust_download_video,
        VideoAdapter,
        TaskManager,
        ZERO_WIDTH_CHARS
    )
except ImportError:
    from ..utils import (
        get_wujiai_host,
        upload_image_wujiai,
        robust_download_video,
        VideoAdapter,
        TaskManager,
        ZERO_WIDTH_CHARS
    )

# 统一使用 TaskManager 接管本地 JSON 数据读写，避免多线程冲突
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

task_manager = TaskManager("wujiai_sora2_history.json")

def _get_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

# --- 节点 1: Sora2 提交任务 ---
class WujiaiSora2SubmitTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True}),
                "model": ("STRING", {"default": "sora-2-all", "multiline": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}),
                "size": (["small", "large"], {"default": "small"}),
                "watermark": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用全局配置"}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "submit"
    CATEGORY = "Nkxx/Wujiai/Sora2"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()
    
    def submit(self, prompt, model, aspect_ratio, duration, size, watermark, api_key="", image=None):
        final_api_key = resolve_api_key(api_key, "wujiai")
        
        if not final_api_key: return ("请配置 API Key",)
        
        try:
            orientation = "landscape" if aspect_ratio == "16:9" else "portrait"
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            
            payload = {
                "model": model,
                "prompt": final_prompt,
                "orientation": orientation,
                "size": size,
                "duration": int(duration),
                "watermark": watermark,
                "images": []
            }
            
            if image is not None:
                img_url = upload_image_wujiai(final_api_key, image)
                if img_url:
                    payload["images"] = [img_url]
                    print(f"[Wujiai Sora2] 图片已上传: {img_url}")
                else:
                    raise Exception("图片上传失败，无法获取网络 URL")

            url = f"{get_wujiai_host()}/v1/video/create"
            res = requests.post(url, headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            
            data = res.json()
            task_id = data.get("id") or data.get("data", {}).get("id")
            
            if not task_id:
                raise Exception(f"API未返回任务ID: {data}")
            
            task_manager.update_task(task_id, {
                "prompt": prompt,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "pending", 
                "model": model
            })
                
            return (f"Sora2 任务提交成功!\n模型: {model}\nID: {task_id}\n请使用 '查询任务状态' 节点刷新。",)
            
        except Exception as e:
            traceback.print_exc()
            return (f"提交失败: {e}",)

# --- 节点 1.5: Sora2 批量提交任务 ---
class WujiaiSora2SubmitBatchTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "model": ("STRING", {"default": "sora-2-all", "multiline": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}),
                "size": (["small", "large"], {"default": "small"}),
                "watermark": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "concurrency": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 999}),
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用全局配置"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_ids", "report")
    FUNCTION = "submit_batch"
    CATEGORY = "Nkxx/Wujiai/Sora2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit_batch(self, file_path, column_name, model, aspect_ratio, duration, size, watermark, concurrency,
                     api_key="", prompt_prefix="", max_count=50, image=None):

        final_api_key = resolve_api_key(api_key, "wujiai")

        if not final_api_key:
            return ("", "API Key 不能为空。")

        if not file_path or not os.path.exists(file_path):
            return ("", "文件路径为空或文件不存在。")

        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return ("", "仅支持 .csv, .xls, .xlsx 文件。")
        except Exception as e:
            return ("", f"读取文件失败: {e}")

        if column_name not in df.columns:
            return ("", f"列 '{column_name}' 不存在。")

        prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[:max_count]]
        if not prompts:
            return ("", f"列 '{column_name}' 中未找到有效 prompt。")

        uploaded_image_url = None
        if image is not None:
            uploaded_image_url = upload_image_wujiai(final_api_key, image)
            if not uploaded_image_url:
                return ("", "图像上传失败，中止批量任务。")

        host = get_wujiai_host()
        headers = _get_headers(final_api_key)
        
        # 统一映射比例
        api_orientation = "portrait" if aspect_ratio == "9:16" else "landscape"

        def submit_task_internal(prompt):
            try:
                final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
                payload = {
                    "model": model.strip(),
                    "prompt": final_prompt,
                    "orientation": api_orientation,
                    "duration": int(duration),
                    "size": size,
                    "watermark": watermark,
                    "images": [uploaded_image_url] if uploaded_image_url else []
                }

                res = requests.post(f"{host}/v1/video/create", headers=headers, json=payload, timeout=60)
                res.raise_for_status()
                submit_data = res.json()

                task_id = submit_data.get("id") or submit_data.get("data", {}).get("id") or submit_data.get("task_id")
                if not task_id:
                    return (prompt, f"API未返回有效ID: {submit_data}")

                task_manager.update_task(task_id, {
                    "prompt": prompt,
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                    "model": model.strip()
                })
                return (prompt, task_id)
            except Exception as e:
                return (prompt, f"提交异常: {str(e)}")

        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            all_results = list(executor.map(submit_task_internal, prompts))

        success_count, fail_count = 0, 0
        output_lines = []

        for _prompt, result in all_results:
            if "异常" in result or "未返回" in result:
                fail_count += 1
            else:
                success_count += 1
            output_lines.append(result)

        report = f"批量提交完成 | 总数: {len(prompts)} | 成功: {success_count} | 失败: {fail_count}"
        return ("\n".join(output_lines), report)

# --- 节点 2: 查询状态 ---
class WujiaiSora2QueryTasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "query"
    CATEGORY = "Nkxx/Wujiai/Sora2"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def query(self, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
            
        if not final_api_key: return ("请配置 API Key",)

        tasks = task_manager.read_tasks()
        pending_tasks = {k: v for k, v in tasks.items() if v.get("status") in task_manager.active_statuses}
        
        for tid, tinfo in pending_tasks.items():
            try:
                query_url = f"{get_wujiai_host()}/v1/video/query?id={tid}"
                res = requests.get(query_url, headers=_get_headers(final_api_key), timeout=30)
                if res.status_code == 200:
                    d = res.json()
                    new_status = d.get("status", "running") 
                    
                    updates = {"status": new_status}
                    if new_status in ["success", "succeeded", "completed"] and d.get("video_url"):
                        updates["status"] = "succeeded"
                        updates["video_url"] = d.get("video_url")
                    elif new_status in ["failed", "failure", "error"]:
                        updates["status"] = "failed"
                        
                    if tasks[tid].get("status") != updates["status"] or "video_url" in updates:
                        task_manager.update_task(tid, updates)
                        
            except Exception as e:
                print(f"[Wujiai Sora2] 查询 {tid} 失败: {e}")
        
        final_tasks = task_manager.read_tasks()
        report = ["--- Wujiai Sora2 任务队列 ---"]
        for tid, info in sorted(final_tasks.items(), key=lambda x: x[1].get('submitted_at', '0'), reverse=True):
            status = info.get("status", "unknown")
            prompt_short = info.get("prompt", "")[:20].replace("\n", "")
            model_info = info.get("model", "sora-2")
            report.append(f"[{status}] ({model_info}) {tid[:8]}... : {prompt_short}...")
            
        return ("\n".join(report),)

# --- 节点 3: 获取视频 ---
class WujiaiSora2GetNextVideo:
    @classmethod
    def INPUT_TYPES(cls): return {}
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "get_video"
    CATEGORY = "Nkxx/Wujiai/Sora2"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_video(self):
        tasks = task_manager.read_tasks()
        target_id = next((k for k, v in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', '0')) if v.get("status") == "succeeded"), None)
        
        if not target_id:
            return (VideoAdapter(None), "没有等待下载的新完成任务。")
            
        url = tasks[target_id].get("video_url")
        output_dir = folder_paths.get_output_directory()
        filename = f"wujiai_sora2_{target_id[:8]}_{uuid.uuid4().hex[:4]}.mp4"
        path = os.path.join(output_dir, filename)
        
        try:
            robust_download_video(url, path)
            task_manager.update_task(target_id, {
                "status": "downloaded",
                "video_path": path
            })
            return (VideoAdapter(path), f"下载成功: {filename}")
        except Exception as e:
            task_manager.update_task(target_id, {"status": "download_failed"})
            return (VideoAdapter(None), f"下载失败: {e}\nURL: {url}")

# --- 注册 ---
NODE_CLASS_MAPPINGS = {
    "WujiaiSora2SubmitTask": WujiaiSora2SubmitTask,
    "WujiaiSora2SubmitBatchTask": WujiaiSora2SubmitBatchTask,
    "WujiaiSora2QueryTasks": WujiaiSora2QueryTasks,
    "WujiaiSora2GetNextVideo": WujiaiSora2GetNextVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WujiaiSora2SubmitTask": "1. Sora2 提交任务 (Wujiai)",
    "WujiaiSora2SubmitBatchTask": "1.5. Sora2 批量提交 (CSV/Excel)",
    "WujiaiSora2QueryTasks": "2. Sora2 查询状态 (Wujiai)",
    "WujiaiSora2GetNextVideo": "3. Sora2 下载视频 (Wujiai)",
}
