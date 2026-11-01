import os
import json
import requests
import base64
import io
import time
from datetime import datetime
import threading
import traceback
import secrets
import concurrent.futures
import re
import pandas as pd
import torch
from PIL import Image

import folder_paths
from ..utils import safe_pil_batch_to_tensor, tensor_to_pil, TaskManager, get_wujiai_host, upload_image_wujiai

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]
universal_task_db = TaskManager("wujiai_universal_async_history.json", max_completed_history=50)
_wujiai_worker_lock = threading.Lock()

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

def sanitize_filename_wujiai(text: str, max_length: int = 100) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', text)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    return sanitized.strip('_')[:max_length]

# ======================================================================
# 辅助函数：提取公共逻辑
# ======================================================================
def _parse_input_images(images):
    """安全解析 ComfyUI 传入的 IMAGE 张量为 PIL 列表"""
    if images is None:
        return []
    image_list = images if isinstance(images, list) else [images]
    pil_images = []
    for img_data in image_list:
        pil_images.extend(tensor_to_pil(img_data))
    return pil_images

def _load_prompts_from_file(file_path: str, column_name: str, prompt_prefix: str = "") -> list[str]:
    """通用的 CSV/Excel 读取与 Prompt 拼接逻辑"""
    if not file_path or not os.path.exists(file_path):
        raise Exception("文件路径为空或文件不存在。")
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise Exception("仅支持 .csv, .xls, .xlsx 文件。")
    
    if column_name not in df.columns:
        raise Exception(f"列 '{column_name}' 不存在。")
    
    prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str)]
    if not prompts:
        raise Exception("未找到有效 prompt。")
    return prompts

# ======================================================================
# 架构核心：大模型 API 策略注册表
# ======================================================================
class BaseProvider:
    def build_request(self, api_key: str, model_name: str, prompt: str, aspect_ratio: str, image_size: str, n: int, subtask_idx: int, pil_images: list, sequential_mode: str = "disabled", max_images: int = 15) -> tuple[str, dict, dict, list | None]:
        raise NotImplementedError

    def parse_response(self, data: dict) -> tuple[list[str] | None, str]:
        raise NotImplementedError

class ProviderRegistry:
    _providers = {}

    @classmethod
    def register(cls, keyword: str, provider_item):
        if isinstance(provider_item, type):
            cls._providers[keyword.lower()] = provider_item()
        else:
            cls._providers[keyword.lower()] = provider_item

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider | None:
        model_name_lower = model_name.strip().lower()
        for kw, provider in cls._providers.items():
            if kw in model_name_lower:
                return provider
        return None

# ======================================================================
# 厂商接入实现
# ======================================================================
class GeminiProvider(BaseProvider):
    def build_request(self, api_key, model_name, prompt, aspect_ratio, image_size, n, subtask_idx, pil_images, sequential_mode="disabled", max_images=15):
        url = f"{get_wujiai_host()}/v1beta/models/{model_name.strip()}:generateContent?key={api_key}"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        if aspect_ratio == "auto":
            if pil_images and len(pil_images) == 1:
                img_ratio = pil_images[0].width / pil_images[0].height
                supported_ratios = ['1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '8:1', '9:16', '16:9', '21:9']
                aspect_ratio = min(supported_ratios, key=lambda r: abs((float(r.split(':')[0]) / float(r.split(':')[1])) - img_ratio))
            else:
                aspect_ratio = "1:1"
        
        base_parts = []
        for pil_img in pil_images:
            buffered = io.BytesIO()
            pil_img.convert("RGB").save(buffered, format="JPEG", quality=90)
            base_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg", 
                    "data": base64.b64encode(buffered.getvalue()).decode("utf-8")
                }
            })

        image_config = {"aspectRatio": aspect_ratio}
        if "2.5" not in model_name.lower():
            image_config["imageSize"] = "1K" if image_size == "默认" else image_size

        base_zwc = "".join(secrets.choice(ZERO_WIDTH_CHARS) for _ in range(secrets.randbelow(4) + 1))
        final_prompt = f"{prompt}{base_zwc}{secrets.choice(ZERO_WIDTH_CHARS) * subtask_idx}"
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": final_prompt}] + base_parts}], 
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": image_config}
        }
        return url, headers, payload, None

    def parse_response(self, data):
        base64_img, returned_text = None, ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "inlineData" in part and "data" in part["inlineData"]:
                    b64_val = part["inlineData"]["data"]
                    base64_img = b64_val.split(",", 1)[-1] if b64_val.startswith("data:image") else b64_val
                    break
                if "text" in part:
                    returned_text += part["text"] + " "
                    
        if base64_img:
            return [base64_img], "" 
            
        finish_reason = candidates[0].get("finishReason", "未知原因") if candidates else "无返回内容"
        error_detail = f"未提取到图像 (FinishReason: {finish_reason})"
        if returned_text.strip():
            error_detail += f" | 模型回复: {returned_text.strip()[:30]}..."
        return None, error_detail

class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def build_request(self, api_key, model_name, prompt, aspect_ratio, image_size, n, subtask_idx, pil_images, sequential_mode="disabled", max_images=15):
        size_map = {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536", "auto": "auto"}
        if aspect_ratio not in size_map:
            raise Exception(f"{self.provider_name} 模型不支持比例 '{aspect_ratio}'。仅支持 1:1, 2:3, 3:2, auto。")
            
        mapped_size = size_map[aspect_ratio]
        base_zwc = "".join(secrets.choice(ZERO_WIDTH_CHARS) for _ in range(secrets.randbelow(4) + 1))
        final_prompt = f"{prompt}{base_zwc}{secrets.choice(ZERO_WIDTH_CHARS) * subtask_idx}"

        if pil_images:
            url = f"{get_wujiai_host()}/v1/images/edits"
            headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
            payload = {"model": model_name.strip(), "prompt": final_prompt, "n": str(n), "size": mapped_size, "response_format": "b64_json"}
            files = []
            for idx, pil_img in enumerate(pil_images[:5]): 
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                files.append(("image", (f"image_{idx+1}.png", buffered.getvalue(), "image/png")))
            return url, headers, payload, files
        else:
            url = f"{get_wujiai_host()}/v1/images/generations"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
            payload = {"model": model_name.strip(), "prompt": final_prompt, "n": n, "size": mapped_size, "response_format": "b64_json"}
            return url, headers, payload, None

    def parse_response(self, data):
        if "data" not in data:
            error_msg = data.get("error", {}).get("message", "未知错误")
            return None, f"{self.provider_name} API 报错: {error_msg} | 完整返回: {str(data)[:200]}"
            
        b64_images = []
        for item in data["data"]:
            if "b64_json" in item:
                b64_val = item["b64_json"]
                b64_images.append(b64_val.split(",", 1)[-1] if b64_val.startswith("data:image") else b64_val)
            elif "url" in item:
                try:
                    res = requests.get(item["url"], timeout=60)
                    res.raise_for_status()
                    b64_images.append(base64.b64encode(res.content).decode("utf-8"))
                except Exception as e:
                    print(f"[{self.provider_name} Provider] 下载 URL 失败: {e}")
                    
        if not b64_images:
            return None, "成功接收响应，但未能提取到图片内容"
        return b64_images, ""

class SeedreamProvider(BaseProvider):
    def build_request(self, api_key, model_name, prompt, aspect_ratio, image_size, n, subtask_idx, pil_images, sequential_mode="disabled", max_images=15):
        if aspect_ratio == "auto":
            if pil_images and len(pil_images) >= 1:
                img_ratio = pil_images[0].width / pil_images[0].height
                supported_ratios = ['1:1', '4:3', '3:4', '16:9', '9:16', '3:2', '2:3', '21:9']
                aspect_ratio = min(supported_ratios, key=lambda r: abs((float(r.split(':')[0]) / float(r.split(':')[1])) - img_ratio))
            else:
                aspect_ratio = "1:1"

        sizes_1k = {"1:1": "1024x1024", "4:3": "1152x864", "3:4": "864x1152", "16:9": "1280x720", "9:16": "720x1280", "3:2": "1248x832", "2:3": "832x1248", "21:9": "1512x648"}
        sizes_2k = {"1:1": "2048x2048", "4:3": "2304x1728", "3:4": "1728x2304", "16:9": "2848x1600", "9:16": "1600x2848", "3:2": "2496x1664", "2:3": "1664x2496", "21:9": "3136x1344"}
        sizes_3k = {"1:1": "3072x3072", "4:3": "3456x2592", "3:4": "2592x3456", "16:9": "4096x2304", "9:16": "2304x4096", "3:2": "3744x2496", "2:3": "2496x3744", "21:9": "4704x2016"}
        sizes_4k = {"1:1": "4096x4096", "4:3": "4704x3520", "3:4": "3520x4704", "16:9": "5504x3040", "9:16": "3040x5504", "3:2": "4992x3328", "2:3": "3328x4992", "21:9": "6240x2656"}
        
        mapped_size = "2048x2048"
        model_name_lower = model_name.strip().lower()
        if "seedream-5" in model_name_lower:
            mapped_size = sizes_3k.get(aspect_ratio, "3072x3072") if image_size == "3K" else sizes_2k.get(aspect_ratio, "2048x2048")
        elif "seedream-4-5" in model_name_lower:
            mapped_size = sizes_4k.get(aspect_ratio, "4096x4096") if image_size == "4K" else sizes_2k.get(aspect_ratio, "2048x2048")
        elif "seedream-4" in model_name_lower:
            if image_size == "1K": mapped_size = sizes_1k.get(aspect_ratio, "1024x1024")
            elif image_size == "4K": mapped_size = sizes_4k.get(aspect_ratio, "4096x4096")
            else: mapped_size = sizes_2k.get(aspect_ratio, "2048x2048")
            
        base_zwc = "".join(secrets.choice(ZERO_WIDTH_CHARS) for _ in range(secrets.randbelow(4) + 1))
        final_prompt = f"{prompt}{base_zwc}{secrets.choice(ZERO_WIDTH_CHARS) * subtask_idx}"

        url = f"{get_wujiai_host()}/v1/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        payload = {
            "model": model_name.strip(), "prompt": final_prompt, "size": mapped_size,
            "watermark": False, "output_format": "png", "response_format": "b64_json",
            "sequential_image_generation": sequential_mode
        }
        
        if sequential_mode == "auto":
            payload["sequential_image_generation_options"] = {"max_images": max_images}
        
        if pil_images:
            img_list = []
            for pil_img in pil_images[:14]: 
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_list.append(f"data:image/png;base64,{b64_str}")
            if img_list:
                payload["image"] = img_list if len(img_list) > 1 else img_list[0]
                
        return url, headers, payload, None

    def parse_response(self, data):
        if "data" not in data:
            return None, f"Seedream API 报错: {data.get('error', {}).get('message', '未知错误')} | 完整返回: {str(data)[:200]}"
        b64_images = []
        for item in data["data"]:
            if "b64_json" in item:
                b64_val = item["b64_json"]
                b64_images.append(b64_val.split(",", 1)[-1] if b64_val.startswith("data:image") else b64_val)
            elif "url" in item:
                try:
                    res = requests.get(item["url"], timeout=60)
                    res.raise_for_status()
                    b64_images.append(base64.b64encode(res.content).decode("utf-8"))
                except Exception as e:
                    print(f"[Seedream Provider] 下载 URL 失败: {e}")
        if not b64_images:
            return None, "成功接收响应，但未能提取到图片内容"
        return b64_images, ""

ProviderRegistry.register("gemini", GeminiProvider)
ProviderRegistry.register("nanobanana", GeminiProvider)
ProviderRegistry.register("grok", OpenAICompatibleProvider("Grok"))
ProviderRegistry.register("gpt", OpenAICompatibleProvider("OpenAI"))
ProviderRegistry.register("seedream", SeedreamProvider)

# ======================================================================
# 后台异步核心 Worker
# ======================================================================
def _background_sync_worker(task_id, subtask_idx, url, headers, payload, files, model_name):
    time.sleep(subtask_idx * 2.0) 
    try:
        print(f"[Wujiai Async] 后台线程启动: {task_id} (子任务 {subtask_idx}) ({model_name})...")
        max_retries = 3
        data = None
        
        session = requests.Session()
        for attempt in range(max_retries):
            with _wujiai_worker_lock:
                current_task = universal_task_db.read_tasks().get(task_id)
                if not current_task or current_task.get("status") == "FAILURE":
                    print(f"[Wujiai Async] 线程终止: 任务 {task_id} 已在外部失效。")
                    session.close()
                    return  

            try:
                req_timeout = (15, 720) 
                if files:
                    res = session.post(url, headers=headers, data=payload, files=files, timeout=req_timeout)
                else:
                    res = session.post(url, headers=headers, json=payload, timeout=req_timeout)
                
                res.raise_for_status()
                data = res.json()
                break
                
            except requests.exceptions.ReadTimeout:
                print(f"\033[93m[Wujiai Async] {task_id}-{subtask_idx} 读取超时 ({attempt+1}/{max_retries})\033[0m")
                if attempt == max_retries - 1: raise Exception("请求重试多次仍读取超时")
                time.sleep(3)
                    
            except Exception as net_e: 
                err_str = str(net_e).split("for url:")[0].strip()
                print(f"[Wujiai Async] {task_id}-{subtask_idx} 请求异常 ({attempt+1}/{max_retries}): {err_str}")
                if attempt == max_retries - 1: raise Exception(f"请求失败: {err_str}")
                time.sleep(5)

        session.close()
        if not data: raise Exception("未获取到返回数据或非 JSON 格式")
        
        provider = ProviderRegistry.get_provider(model_name)
        b64_img_list, error_detail = provider.parse_response(data)
        
        with _wujiai_worker_lock:
            task = universal_task_db.read_tasks().get(task_id)
            if not task or task.get("status") == "FAILURE": return

            if b64_img_list:
                temp_dir = os.path.join(folder_paths.get_temp_directory(), "wujiai_async_cache")
                os.makedirs(temp_dir, exist_ok=True)
                cache_file = os.path.join(temp_dir, f"{task_id}_{subtask_idx}.txt")
                with open(cache_file, "w") as f:
                    json.dump(b64_img_list, f)
                task["subtasks"][subtask_idx]["status"] = "SUCCESS"
                task["subtasks"][subtask_idx]["cache_file"] = cache_file
                print(f"[Wujiai Async] {task_id} ({subtask_idx}) 出图成功！共 {len(b64_img_list)} 张。")
            else:
                task["subtasks"][subtask_idx]["status"] = "FAILURE"
                task["subtasks"][subtask_idx]["failReason"] = error_detail
                
            _check_and_update_task_status(task_id, task)
            
    except Exception as e:
        err_str = str(e).split("for url:")[0].strip()
        print(f"\n[Wujiai 错误] {task_id}-{subtask_idx} 失败: {err_str}\n")
        with _wujiai_worker_lock:
            task = universal_task_db.read_tasks().get(task_id)
            if task and task.get("status") != "FAILURE":
                task["subtasks"][subtask_idx]["status"] = "FAILURE"
                task["subtasks"][subtask_idx]["failReason"] = err_str
                _check_and_update_task_status(task_id, task)

def _check_and_update_task_status(task_id, task):
    all_done = True
    any_success = False
    for sub in task["subtasks"]:
        if sub["status"] == "running":
            all_done = False
            break
        if sub["status"] == "SUCCESS":
            any_success = True
    
    if all_done:
        task["status"] = "SUCCESS" if any_success else "FAILURE"
    universal_task_db.update_task(task_id, task)

def _launch_async_tasks(task_type: str, prompt_desc: str, model_name: str, aspect_ratio: str, image_size: str, sequential_mode: str, max_images: int, n: int, prompts: list, pil_images_list: list, final_api_key: str, provider: BaseProvider):
    task_id = f"WUJIAI_{task_type}_{int(time.time()*1000)}"
    concurrency = len(prompts)
    
    with _wujiai_worker_lock:
        tasks_dict = universal_task_db.read_tasks()
        max_num = max([t.get("task_number", 0) for t in tasks_dict.values() if isinstance(t, dict)], default=0)
        task_number = max_num + 1

        subtasks = [{"index": i, "status": "running", "cache_file": None, "failReason": None, "original_prompt": p} for i, p in enumerate(prompts)]
        universal_task_db.update_task(task_id, {
            "type": task_type, "model": model_name, "prompt": prompt_desc, 
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running",
            "concurrency": concurrency, "n": n, "subtasks": subtasks, "task_number": task_number
        })
    
    for i, p_text in enumerate(prompts):
        try:
            url, headers, payload, files = provider.build_request(
                final_api_key, model_name, p_text, aspect_ratio, image_size, n, i, pil_images_list, sequential_mode, max_images
            )
            threading.Thread(target=_background_sync_worker, args=(task_id, i, url, headers, payload, files, model_name), daemon=True).start()
        except Exception as build_err:
            with _wujiai_worker_lock:
                task = universal_task_db.read_tasks().get(task_id)
                task["subtasks"][i]["status"] = "FAILURE"
                task["subtasks"][i]["failReason"] = str(build_err)
                _check_and_update_task_status(task_id, task)
                
    return task_number, concurrency

# ======================================================================
# 节点基类：统一报错回传逻辑
# ======================================================================
class _UniversalNodeBaseWujiai:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
        
    def _create_error_result(self, error_message: str, return_tensor: bool = True):
        print(f"[Wujiai 错误] {error_message}")
        if return_tensor:
            return (torch.zeros((1, 1, 1, 3), dtype=torch.float32), f"失败: {error_message}")
        return {"ui": {"string": [f"失败: {error_message}"]}, "result": (f"失败: {error_message}",)}

# ======================================================================
# 节点 1: 通用生图 (同步版)
# ======================================================================
class UniversalSyncGeneratorWujiai(_UniversalNodeBaseWujiai):
    CATEGORY = "Nkxx/Wujiai/Universal"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        # 严格保持最初的版本字典顺序
        return {
            "required": {
                "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cyberpunk cat"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "3:4", "4:3", "2:3", "3:2", "auto", "21:9"], {"default": "1:1"}),
                "image_size": (["默认", "1K", "2K", "3K", "4K"], {"default": "默认"}),
                "sequential_mode": (["disabled", "auto"], {"default": "disabled", "tooltip": "组图模式"}),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15, "step": 1}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {"images": ("IMAGE",), "api_key": ("STRING", {"default": ""})}
        }

    def generate(self, model_name, prompt, aspect_ratio, image_size, sequential_mode, max_images, n, concurrency, images=None, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return self._create_error_result("请配置 API Key")

        provider = ProviderRegistry.get_provider(model_name)
        if not provider: return self._create_error_result(f"暂未适配模型: {model_name}")

        pil_images_list = _parse_input_images(images)

        def _fetch_single_image(i):
            url, headers, payload, files = provider.build_request(
                final_api_key, model_name, prompt, aspect_ratio, image_size, n, i, pil_images_list, sequential_mode, max_images
            )
            
            # 增加 3 次重试机制
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    if files:
                        res = requests.post(url, headers=headers, data=payload, files=files, timeout=(15, 720))
                    else:
                        res = requests.post(url, headers=headers, json=payload, timeout=(15, 720))
                        
                    res.raise_for_status()
                    
                    b64_img_list, err = provider.parse_response(res.json())
                    if not b64_img_list: 
                        raise Exception(err)
                        
                    return [Image.open(io.BytesIO(base64.b64decode(b64_str))) for b64_str in b64_img_list]
                    
                except requests.exceptions.RequestException as e:
                    # 捕获网络和 HTTP 错误
                    last_error = str(e).split('for url:')[0].strip()
                    print(f"[Wujiai Sync] 节点请求失败 ({attempt+1}/{max_retries})，等待 3 秒后重试... 错误: {last_error}")
                    time.sleep(3)
                except Exception as e:
                    # 捕获其他解析错误
                    last_error = str(e)
                    print(f"[Wujiai Sync] 节点解析失败 ({attempt+1}/{max_retries})，等待 3 秒后重试... 错误: {last_error}")
                    time.sleep(3)
                    
            # 如果 3 次都失败了，再向上抛出异常
            raise Exception(f"请求重试 {max_retries} 次后最终失败: {last_error}")

        pil_images, errors = [], []
        print(f"[Wujiai Sync] 开始同步并发生成，并发数: {concurrency}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {executor.submit(_fetch_single_image, i): i for i in range(concurrency)}
            for future in concurrent.futures.as_completed(future_to_idx):
                try: 
                    pil_images.extend(future.result()) 
                except Exception as e: 
                    errors.append(str(e).split('for url:')[0].strip())
                    
        if not pil_images:
            return self._create_error_result(f"全部出图失败: {errors[0] if errors else '未知错误'}")

        report_msg = f"{model_name} | 成功生成: {len(pil_images)} 张"
        if errors:
            report_msg += f" | 失败: {len(errors)} 次"
            print(f"[Wujiai Sync] ⚠️ 并发生成部分请求失败: {errors}")

        return (safe_pil_batch_to_tensor(pil_images), report_msg)

# ======================================================================
# 节点 2: 通用异步提交
# ======================================================================
class UniversalAsyncSubmitWujiai(_UniversalNodeBaseWujiai):
    CATEGORY = "Nkxx/Wujiai/Universal"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "submit"
    OUTPUT_NODE = True 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cyberpunk cat"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "3:4", "4:3", "2:3", "3:2", "auto", "21:9"], {"default": "1:1"}),
                "image_size": (["默认", "1K", "2K", "3K", "4K"], {"default": "默认"}),
                "sequential_mode": (["disabled", "auto"], {"default": "disabled", "tooltip": "组图模式"}),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15, "step": 1}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {"images": ("IMAGE",), "api_key": ("STRING", {"default": ""})}
        }

    def submit(self, model_name, prompt, aspect_ratio, image_size, sequential_mode, max_images, n, concurrency, images=None, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return self._create_error_result("缺少 API Key", return_tensor=False)

        provider = ProviderRegistry.get_provider(model_name)
        if not provider: return self._create_error_result(f"未适配模型 {model_name}", return_tensor=False)

        try:
            # === 新增：安全熔断机制（链式间隔检测） ===
            with _wujiai_worker_lock:
                tasks = universal_task_db.read_tasks()

                # 1. 筛选出所有未下载的活跃任务
                active_tasks = [t for t in tasks.values() if t.get("status") != "DOWNLOADED"]
                active_tasks.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)

                identical_count = 0
                last_time = datetime.now()

                # 2. 统计极短时间内的连续同提示词提交
                for t in active_tasks:
                    if t.get("type") == "ASYNC" and t.get("prompt") == prompt:
                        try:
                            t_time = datetime.strptime(t.get("submitted_at", ""), "%Y-%m-%d %H:%M:%S")
                            if (last_time - t_time).total_seconds() <= 15:
                                identical_count += 1
                                last_time = t_time
                            else:
                                break
                        except Exception:
                            break
                    else:
                        break

                # 3. 触发熔断拦截
                if identical_count >= 5:
                    error_msg = (
                        f"【安全熔断触发】检测到系统正在自动连续提交相同的提示词 (已拦截第 {identical_count + 1} 次)！\n"
                        f"为防止您的额度因误操作被异常消耗，当前提交已被强制阻断。\n"
                        f"💡 您的前 {identical_count} 次任务已成功发送，请放心。\n"
                        f"请检查并清理 ComfyUI 队列，或关闭自动排队 (Auto Queue)。"
                    )
                    raise ValueError(error_msg)
            # ==========================================

            task_num, _ = _launch_async_tasks(
                "ASYNC", prompt, model_name, aspect_ratio, image_size, sequential_mode, max_images, n, 
                [prompt]*concurrency, _parse_input_images(images), final_api_key, provider
            )
            
            msg = f"任务提交成功 | 任务{task_num} | 模型: {model_name} | 子任务数: {concurrency}" + (f" (x{n})" if n > 1 else "")
            return {"ui": {"string": [msg]}, "result": (msg,)}
        
        except Exception as e:
            # 捕获我们自定义抛出的熔断警告并直接向上抛出，以触发红框报错打断 ComfyUI 自动运行
            if isinstance(e, ValueError) and "【安全熔断触发】" in str(e):
                raise e
            return self._create_error_result(f"提交失败: {e}", return_tensor=False)

# ======================================================================
# 节点 3: 通用异步查询下载
# ======================================================================
class UniversalAsyncQueryWujiai(_UniversalNodeBaseWujiai):
    CATEGORY = "Nkxx/Wujiai/Universal"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "query_and_download"

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": ""})}}

    def _generate_display_status(self, tasks: dict) -> str:
        if not tasks: return "--- 当前无任务记录 ---"
        status_map = {"running": "运行中", "SUCCESS": "待提取", "DOWNLOADED": "已下载", "FAILURE": "失败"}
        lines = ["--- 通用异步任务队列 ---"]
        
        for tid, tinfo in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
            raw_status = tinfo.get('status', 'running')
            display_str = status_map.get(raw_status, raw_status)
            task_num = tinfo.get('task_number', '?')
            model_name = tinfo.get('model', '未知模型') # 提取模型名称
            
            # 提示词清理并保留前15个字符
            prompt_clean = tinfo.get('prompt', '').replace('\n', ' ')
            prompt_snippet = prompt_clean[:15] + "..." if len(prompt_clean) > 15 else prompt_clean
            
            if "subtasks" in tinfo:
                success_sub = sum(1 for s in tinfo["subtasks"] if s.get("status") == "SUCCESS")
                fail_sub = sum(1 for s in tinfo["subtasks"] if s.get("status") == "FAILURE")
                info = f"[{display_str}] 任务{task_num} ({prompt_snippet}) | {model_name} | {len(tinfo['subtasks'])}个子任务"
                if fail_sub > 0: info += f" (成: {success_sub} | 败: {fail_sub})"
                lines.append(info)
            else:
                reason = f" | 理由: {tinfo.get('failReason', '未知')}" if raw_status == "FAILURE" else ""
                lines.append(f"[{display_str}] 任务{task_num} ({prompt_snippet}) | {model_name}{reason}")
        return "\n".join(lines)

    def query_and_download(self, api_key=""):
        with _wujiai_worker_lock:
            tasks = universal_task_db.read_tasks()
            now = time.time()
            changed = False
            for tid, tinfo in tasks.items():
                if tinfo.get("status") == "running" and tinfo.get("submitted_at"):
                    try:
                        if now - time.mktime(time.strptime(tinfo["submitted_at"], "%Y-%m-%d %H:%M:%S")) > 900:  
                            tinfo["status"] = "FAILURE"
                            tinfo["failReason"] = "任务超时 (超15分钟)"
                            for sub in tinfo.get("subtasks", []):
                                if sub.get("status") == "running":
                                    sub["status"] = "FAILURE"
                                    sub["failReason"] = "执行超时"
                            universal_task_db.update_task(tid, tinfo)
                            changed = True
                    except: pass
            if changed: tasks = universal_task_db.read_tasks()

        if not tasks: return {"ui": {"string": ["当前无任务"]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), "无任务")}
        panel_status = self._generate_display_status(tasks)

        target_id = next((k for k, v in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', '')) if v.get("status") == "SUCCESS"), None)
        if not target_id: return {"ui": {"string": [f"暂无可用结果\n\n{panel_status}"]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), "暂无结果")}

        task_info = tasks[target_id]
        try:
            pil_images, files_to_delete = [], []
            
            def extract_images_from_cache(c_file):
                with open(c_file, "r") as f: raw_data = f.read()
                try: b64_list = json.loads(raw_data) if isinstance(json.loads(raw_data), list) else [raw_data]
                except: b64_list = [raw_data] 
                for b64_str in b64_list:
                    pil_images.append(Image.open(io.BytesIO(base64.b64decode(b64_str))))
                files_to_delete.append(c_file)

            if "subtasks" in task_info:
                for sub in task_info["subtasks"]:
                    if sub.get("status") == "SUCCESS" and sub.get("cache_file") and os.path.exists(sub["cache_file"]):
                        extract_images_from_cache(sub["cache_file"])
            elif task_info.get("cache_file") and os.path.exists(task_info["cache_file"]):
                extract_images_from_cache(task_info["cache_file"])
            
            if not pil_images: raise Exception("解析图片失败或本地所有缓存文件均不存在")
            
            for f_path in files_to_delete:
                try: os.remove(f_path)
                except: pass
            
            with _wujiai_worker_lock:
                universal_task_db.update_task(target_id, {"status": "DOWNLOADED"})
                final_panel = self._generate_display_status(universal_task_db.read_tasks())
            
            return {"ui": {"string": [final_panel]}, "result": (safe_pil_batch_to_tensor(pil_images), final_panel)}
            
        except Exception as e:
            with _wujiai_worker_lock:
                universal_task_db.update_task(target_id, {"status": "FAILURE", "failReason": str(e)})
            return {"ui": {"string": [f"提取失败: {e}\n\n{panel_status}"]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), "提取失败")}

# ======================================================================
# 新增节点 4: 通用异步批量提交
# ======================================================================
class UniversalAsyncBatchSubmitWujiai(_UniversalNodeBaseWujiai):
    CATEGORY = "Nkxx/Wujiai/Universal"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "submit_batch"
    OUTPUT_NODE = True 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "3:4", "4:3", "2:3", "3:2", "auto", "21:9"], {"default": "1:1"}),
                "image_size": (["默认", "1K", "2K", "3K", "4K"], {"default": "默认"}),
                "sequential_mode": (["disabled", "auto"], {"default": "disabled", "tooltip": "组图模式"}),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15, "step": 1}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "label": "单提示词执行次数"}),
            },
            "optional": {"images": ("IMAGE",), "api_key": ("STRING", {"default": ""})}
        }

    def submit_batch(self, file_path, column_name, prompt_prefix, model_name, aspect_ratio, image_size, sequential_mode, max_images, n, executions_per_prompt, images=None, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return self._create_error_result("缺少 API Key", return_tensor=False)

        provider = ProviderRegistry.get_provider(model_name)
        if not provider: return self._create_error_result(f"未适配模型 {model_name}", return_tensor=False)

        try:
            base_prompts = _load_prompts_from_file(file_path, column_name, prompt_prefix)
            prompts = [p for p in base_prompts for _ in range(max(1, executions_per_prompt))]
        except Exception as e:
            return self._create_error_result(str(e), return_tensor=False)

        task_num, total_tasks = _launch_async_tasks(
            "BATCH", f"批量文件: {os.path.basename(file_path)}", model_name, aspect_ratio, image_size, sequential_mode, max_images, n, 
            prompts, _parse_input_images(images), final_api_key, provider
        )
        
        msg = f"批量提交完成 | 任务{task_num} | 模型: {model_name} | 总提交: {total_tasks}" + (f" (x{n})" if n > 1 else "")
        return {"ui": {"string": [msg]}, "result": (msg,)}

# ======================================================================
# 新增节点 5: 通用同步文件夹批量处理
# ======================================================================
class UniversalBatchDirWujiai(_UniversalNodeBaseWujiai):
    CATEGORY = "Nkxx/Wujiai/Universal"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_preview_15", "status_report")
    FUNCTION = "execute_batch_dir"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
                "directory_path": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "csv_file_path": ("STRING", {"default": "", "placeholder": "填写 CSV/Excel 文件路径至此"}),
                "column_name": ("STRING", {"default": "prompt", "label": "CSV列名"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "3:4", "4:3", "2:3", "3:2", "auto", "21:9"], {"default": "1:1"}),
                "image_size": (["默认", "1K", "2K", "3K", "4K"], {"default": "默认"}),
                "sequential_mode": (["disabled", "auto"], {"default": "disabled", "tooltip": "组图模式"}),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15, "step": 1}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_concurrent_files": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "fixed_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "备用：如果不用CSV，则在此填提示词跑所有图"}),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    def _process_and_save_single_file(self, image_idx, file_path, prompts, output_dir, model_name, aspect_ratio, image_size, n, sequential_mode, max_images, final_api_key, need_preview, provider):
        if os.path.getsize(file_path) == 0:
            return 0, [], [f"文件损坏: {os.path.basename(file_path)}"]

        base_filename = sanitize_filename_wujiai(os.path.splitext(os.path.basename(file_path))[0]) 
        try:
            with Image.open(file_path) as img:
                img.verify()
                img = Image.open(file_path) 
                img_rgb = img.convert("RGB")
                
            inner_workers = min(len(prompts), 5) 
            file_success_count, file_pils, file_errors = 0, [], []

            def strict_task_with_retry(prompt_idx, p_text):
                last_error = ""
                for attempt in range(3):
                    try:
                        url, headers, payload, files = provider.build_request(
                            final_api_key, model_name, p_text, aspect_ratio, image_size, n, prompt_idx, [img_rgb], sequential_mode, max_images
                        )
                        res = requests.post(url, headers=headers, data=payload, files=files, timeout=(15, 720)) if files else requests.post(url, headers=headers, json=payload, timeout=(15, 720))
                        res.raise_for_status()
                        
                        b64_img_list, err = provider.parse_response(res.json())
                        if not b64_img_list:
                            last_error = err
                            time.sleep(1)
                            continue

                        pils = [Image.open(io.BytesIO(base64.b64decode(b64_str))) for b64_str in b64_img_list]
                        saved_local, saved_pils = 0, []
                        
                        for j, out_img in enumerate(pils):
                            prefix = f"Img{image_idx+1:03d}_{base_filename}"
                            suffix = f"P{prompt_idx+1:03d}_{j+1}" if len(pils) > 1 else f"P{prompt_idx+1:03d}" 
                            save_name = f"{prefix}_{suffix}.png"
                            save_path = os.path.join(output_dir, save_name)
                            
                            counter = 1
                            while os.path.exists(save_path):
                                save_path = os.path.join(output_dir, f"{prefix}_{suffix}_{counter}.png")
                                counter += 1
                            
                            try:
                                out_img.save(save_path, "PNG", compress_level=4)
                                saved_local += 1
                                saved_pils.append(out_img)
                            except Exception as save_err:
                                print(f"保存失败: {save_err}")

                        return saved_local, saved_pils, [] 
                    except Exception as e:
                        last_error = str(e)
                        time.sleep(1)
                
                return 0, [], [f"RetryFailed: {last_error}"]

            with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as inner_exc:
                futures = [inner_exc.submit(strict_task_with_retry, i, p) for i, p in enumerate(prompts)]
                for f in concurrent.futures.as_completed(futures):
                    cnt, ret_pils, errs = f.result()
                    file_success_count += cnt
                    if errs: file_errors.append(errs[0])
                    if need_preview and len(file_pils) < 15:
                        file_pils.extend(ret_pils)

            if file_success_count == 0 and file_errors:
                return 0, [], [f"{base_filename}: {file_errors[0]}"]
            return file_success_count, file_pils, []

        except Exception as e:
            traceback.print_exc()
            return 0, [], [f"{base_filename} 异常: {str(e)[:50]}"]

    def execute_batch_dir(self, directory_path, csv_file_path, column_name, model_name, max_concurrent_files, aspect_ratio, image_size, sequential_mode, max_images, n, executions_per_prompt, fixed_prompt="", api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return self._create_error_result("未配置 API Key")

        provider = ProviderRegistry.get_provider(model_name)
        if not provider: return self._create_error_result(f"未适配模型 {model_name}")

        if not directory_path or not os.path.exists(directory_path):
            return self._create_error_result("文件夹路径不存在")

        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in valid_exts])
        if not image_files: return self._create_error_result("文件夹内无有效图片")

        source_mode = "CSV"
        try:
            base_prompts = _load_prompts_from_file(csv_file_path, column_name)
        except:
            if not fixed_prompt.strip():
                return self._create_error_result("必须提供有效 CSV 或 固定提示词。")
            base_prompts = [fixed_prompt]
            source_mode = "Fixed Prompt"

        final_prompts = [p for p in base_prompts for _ in range(executions_per_prompt)]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(folder_paths.get_output_directory(), f"Wujiai_Batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)

        failed_list, total_success_imgs, preview_pil_images = [], 0, []
        all_file_logs = [f"=== Wujiai Batch Log ({timestamp}) ===", f"Source: {source_mode}, Prompts: {len(base_prompts)}, Images: {len(image_files)}"]
        
        consecutive_failures = 0
        MAX_FAILURES = 10 
        ABORT_FLAG = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            futures_map = {} 
            for idx, f in enumerate(image_files):
                if ABORT_FLAG: break
                future = executor.submit(
                    self._process_and_save_single_file, 
                    idx, f, final_prompts, batch_output_dir, model_name, aspect_ratio, image_size, n, sequential_mode, max_images, final_api_key, len(preview_pil_images) < 15, provider
                )
                futures_map[future] = os.path.basename(f)
            
            for future in concurrent.futures.as_completed(futures_map):
                fname = futures_map[future]
                if ABORT_FLAG:
                    future.cancel()
                    continue

                try:
                    count, pils, errs = future.result()
                    if count > 0:
                        total_success_imgs += count
                        consecutive_failures = 0 
                        all_file_logs.append(f"[√] {fname} 生成 {count} 张")
                        if pils and len(preview_pil_images) < 15:
                            preview_pil_images.extend(pils[:15 - len(preview_pil_images)])
                    else:
                        consecutive_failures += 1
                        err_msg = errs[0] if errs else "0 output"
                        all_file_logs.append(f"[X] {fname} {err_msg}")
                        failed_list.append(f"{fname} -> {err_msg}")
                        
                        if consecutive_failures >= MAX_FAILURES:
                            ABORT_FLAG = True
                            fail_msg = "🚨 触发熔断保护：连续失败过多次，停止后续任务。"
                            all_file_logs.append(fail_msg)
                            failed_list.append("BATCH ABORTED (Circuit Breaker)")
                            for f_pending in futures_map:
                                f_pending.cancel()
                            
                except Exception as e:
                    failed_list.append(f"{fname} -> 错误: {str(e)[:50]}")

        ui_summary = [
            f"✅ 批量任务完成" if not ABORT_FLAG else "❌ 任务异常终止 (熔断)",
            f"📁 输入: {os.path.basename(directory_path)}",
            f"🔢 总任务: {len(image_files) * len(final_prompts)}",
            f"🖼️ 成功: {total_success_imgs}",
            f"❌ 失败: {len(failed_list)}",
            f"💾 路径: {batch_output_dir}"
        ]
        
        if failed_list:
            ui_summary.append("\n--- 失败记录 ---")
            for fail in failed_list[:5]: ui_summary.append(f"• {fail}")
            with open(os.path.join(batch_output_dir, "batch_error_log.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(all_file_logs) + "\n\n=== 失败详情 ===\n" + "\n".join(failed_list))
        else:
            ui_summary.append("\n✨ 完美！所有文件均处理成功。")
        
        final_ui_text = "\n".join(ui_summary)
        final_tensor = safe_pil_batch_to_tensor(preview_pil_images) if preview_pil_images else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"string": [final_ui_text]}, "result": (final_tensor, final_ui_text)}


# ======================================================================
# 新增节点 6: Wujiai LLM/VLM Writer (生成 Prompt / 图文问答)
# ======================================================================
class WujiaiLLMWriter(_UniversalNodeBaseWujiai):
    CATEGORY = "Nkxx/Wujiai/语言模型"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gemini-3-flash-preview"}),
                "main_prompt": ("STRING", {"multiline": True, "default": "请为我生成5条关于“夏日海滩”的Midjourney绘画prompt"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "output_filename": ("STRING", {"default": "wujiai_generated_prompts.csv"}),
                "column_name": ("STRING", {"default": "prompt"}),
            }, 
            "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    FUNCTION = "execute"

    def execute(self, model_name, main_prompt, system_prompt, output_filename, column_name, images=None, api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: return self._create_error_result("未配置 Wujiai API Key。", return_tensor=False)
        
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        
        pil_images = _parse_input_images(images)
        
        if pil_images:
            user_content = [{"type": "text", "text": main_prompt.strip()}]
            for img in pil_images:
                uploaded_url = upload_image_wujiai(final_api_key, img)
                if uploaded_url:
                    user_content.append({
                        "type": "image_url", 
                        "image_url": {"url": uploaded_url}
                    })
                else:
                    return self._create_error_result("图片上传失败，请检查网络或 API Key。", return_tensor=False)
            
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": main_prompt.strip()})
        
        url = f"{get_wujiai_host()}/v1/chat/completions"
        headers = {
           'Accept': 'application/json',
           'Authorization': f'Bearer {final_api_key}',
           'Content-Type': 'application/json'
        }
        payload = {
           "model": model_name.strip(),
           "messages": messages
        }
        
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=180)
            res.raise_for_status()
            llm_response = res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return self._create_error_result(f"API调用失败: {e} \n返回内容: {res.text if 'res' in locals() else ''}", return_tensor=False)
        
        try:
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            parsed_list = json.loads(json_match.group()) if json_match else [l.strip() for l in llm_response.split('\n') if l.strip()]
        except (json.JSONDecodeError, TypeError):
             parsed_list = [line.strip() for line in llm_response.split('\n') if llm_response.strip()]

        parsed_list = [item for item in parsed_list if item]
        if not parsed_list or not isinstance(parsed_list, list): 
            return self._create_error_result(f"LLM未返回有效列表。收到: {llm_response}", return_tensor=False)
        
        df = pd.DataFrame(parsed_list, columns=[column_name.strip()])
        
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, output_filename.strip())

        if full_path.lower().endswith('.csv'): df.to_csv(full_path, index=False, encoding='utf-8-sig')
        elif full_path.lower().endswith(('.xls', '.xlsx')): df.to_excel(full_path, index=False)
        else: return self._create_error_result("文件名必须以 .csv, .xls, 或 .xlsx 结尾。", return_tensor=False)
        
        status = f"成功生成 {len(parsed_list)} 条记录并写入: {output_filename}"
        return {"ui": {"string": [status]}, "result": (full_path, status)}

# ======================================================================
# 节点注册
# ======================================================================
NODE_CLASS_MAPPINGS = {
    "UniversalSyncGeneratorWujiai": UniversalSyncGeneratorWujiai,
    "UniversalAsyncSubmitWujiai": UniversalAsyncSubmitWujiai,
    "UniversalAsyncQueryWujiai": UniversalAsyncQueryWujiai,
    "UniversalAsyncBatchSubmitWujiai": UniversalAsyncBatchSubmitWujiai,
    "UniversalBatchDirWujiai": UniversalBatchDirWujiai,
    "WujiaiLLMWriter": WujiaiLLMWriter                   
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSyncGeneratorWujiai": "Wujiai 通用生图 (同步)",
    "UniversalAsyncSubmitWujiai": "Wujiai 通用生图提交 (异步)",
    "UniversalAsyncQueryWujiai": "Wujiai 通用查询下载 (异步)",
    "UniversalAsyncBatchSubmitWujiai": "Wujiai 通用批量提交 (异步 CSV/Excel)", 
    "UniversalBatchDirWujiai": "Wujiai 通用文件夹批量处理 (CSV/Excel)",
    "WujiaiLLMWriter": "✍️ Wujiai LLM/VLM Writer"
}
