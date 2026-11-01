# grsai_nodes/gpt_image_nodes_grsai.py
import os
import json
import concurrent.futures
import traceback
import time
import requests
import torch
import pandas as pd
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image


# 从公共工具箱导入需要的核心函数
from ..utils import (
    TaskManager,
    download_image,
    pil_to_tensor,
    safe_pil_batch_to_tensor,
    tensor_to_pil,
    upload_image_grsai
)

# --- API 客户端 (专门处理 GPT Image 特殊的 SSE 数据流) ---
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

class GrsaiAPIError(Exception): pass

GPT_IMAGE_ASPECT_RATIOS = [
    "auto", "1:1", "3:2", "2:3", "16:9", "9:16", "5:4", "4:5",
    "4:3", "3:4", "21:9", "9:21", "1:3", "3:1", "2:1", "1:2"
]
GPT_IMAGE_STANDARD_MODELS = ["gpt-image-2", "gpt-image-2.5"]
GPT_IMAGE_VIP_MODELS = ["gpt-image-2-vip", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]
GPT_IMAGE_MODEL_CHOICES = GPT_IMAGE_STANDARD_MODELS + GPT_IMAGE_VIP_MODELS

GPT_IMAGE_VIP_SIZE_MAP = {
    "1:1": {"1k": "1248x1248", "2k": "2048x2048", "4k": "2880x2880"},
    "3:2": {"1k": "1536x1024", "2k": "2496x1664", "4k": "3504x2336"},
    "2:3": {"1k": "1024x1536", "2k": "1664x2496", "4k": "2336x3504"},
    "16:9": {"1k": "1792x1008", "2k": "2816x1584", "4k": "3840x2160"},
    "9:16": {"1k": "1008x1792", "2k": "1584x2816", "4k": "2160x3840"},
    "4:3": {"1k": "1472x1104", "2k": "2368x1776", "4k": "3264x2448"},
    "3:4": {"1k": "1104x1472", "2k": "1776x2368", "4k": "2448x3264"},
    "5:4": {"1k": "1440x1152", "2k": "2320x1856", "4k": "3200x2560"},
    "4:5": {"1k": "1152x1440", "2k": "1856x2320", "4k": "2560x3200"},
    "21:9": {"1k": "2016x864", "2k": "3024x1296", "4k": "3696x1584"},
    "9:21": {"1k": "864x2016", "2k": "1296x3024", "4k": "1584x3696"},
    "1:3": {"1k": "720x2160", "2k": "1184x3552", "4k": "1280x3840"},
    "3:1": {"1k": "2160x720", "2k": "3552x1184", "4k": "3840x1280"},
    "2:1": {"1k": "1760x880", "2k": "2912x1456", "4k": "3840x1920"},
    "1:2": {"1k": "880x1760", "2k": "1456x2912", "4k": "1920x3840"},
}

class GrsaiGPTAPI:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip(): 
            raise GrsaiAPIError("API密钥不能为空")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json; charset=utf-8", 
            "User-Agent": "ComfyUI-GptImage/2.0", 
            "Authorization": f"Bearer {self.api_key}"
        })
        self.base_url = "https://grsai.dakka.com.cn"

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, timeout: int = 500) -> Dict[str, Any]:
        """专门处理 GPT Image 接口可能返回的 SSE (data: ...) 格式流"""
        url = f"{self.base_url}{endpoint}"
        response = None
        try:
            response = self.session.request(method, url, json=data, timeout=timeout)
            response.raise_for_status()
            raw_text = response.text.strip()
            
            # 1. 尝试作为标准 JSON 解析
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                pass 

            # 2. 尝试按 SSE 流处理 (提取最后一条有效 JSON)
            lines = raw_text.split('\n')
            last_valid_json = None
            for line in lines:
                line = line.strip()
                if line.startswith("data:"):
                    try:
                        json_str = line[5:].strip()
                        if json_str == "[DONE]" or not json_str: continue 
                        last_valid_json = json.loads(json_str)
                    except:
                        continue
            
            if last_valid_json:
                return last_valid_json
                
            raise GrsaiAPIError(f"无法解析API响应数据: {raw_text[:200]}...")

        except requests.exceptions.RequestException as e:
            msg = str(e)
            if response is not None: msg += f" | Response: {response.text[:100]}"
            raise GrsaiAPIError(f"网络请求失败: {msg}")

    def _build_gpt_image_payload(
        self,
        prompt: str,
        urls: List[str],
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        variants: int,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "urls": urls,
            "shutProgress": True
        }

        if model in GPT_IMAGE_VIP_MODELS:
            if quality and quality.strip():
                payload["quality"] = quality.strip()
            if aspect_ratio == "auto":
                raise GrsaiAPIError(f"{model} does not support auto aspect ratio; choose a concrete ratio.")
            resolution_key = "2k" if resolution == "默认" else resolution.lower()
            if not resolution or resolution in ("默认", "榛樿"):
                resolution_key = "2k"
            mapped_size = GPT_IMAGE_VIP_SIZE_MAP.get(aspect_ratio, {}).get(resolution_key)
            if not mapped_size:
                raise GrsaiAPIError(f"Unsupported VIP aspect/resolution: {aspect_ratio} / {resolution}")
            payload["aspectRatio"] = mapped_size
        else:
            payload["aspectRatio"] = aspect_ratio

        return payload

    def gpt_image_generate(
        self,
        prompt: str,
        urls: List[str],
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        variants: int,
    ) -> Tuple[List[Image.Image], List[str]]:
        payload = self._build_gpt_image_payload(prompt, urls, model, aspect_ratio, resolution, quality, variants)

        data = self._make_request("POST", "/v1/draw/completions", data=payload, timeout=500)

        # 错误检查
        if isinstance(data, dict):
            if data.get("code") and data.get("code") != 0:
                 raise GrsaiAPIError(f"API 业务错误: {data.get('msg')} (Code: {data.get('code')})")
            if data.get("status") == "failed":
                 raise GrsaiAPIError(f"生成任务失败: {data.get('failure_reason', '未知原因')}")

        # 解析结果图片 URL (兼容新旧接口结构)
        results_info = []
        if "results" in data and isinstance(data["results"], list):
            results_info = data["results"]
        elif "data" in data and isinstance(data["data"], dict) and "results" in data["data"]:
            results_info = data["data"]["results"]
        elif "url" in data and isinstance(data["url"], str) and data["url"]:
            results_info = [{"url": data["url"]}]
        elif "data" in data and isinstance(data["data"], dict) and "url" in data["data"] and data["data"]["url"]:
            results_info = [{"url": data["data"]["url"]}]

        if not results_info:
             raise GrsaiAPIError("API响应成功但未找到图片URL，请检查控制台日志。")

        target_urls = [r["url"] for r in results_info if "url" in r and r["url"]]
        
        # 多线程下载
        pil_images, download_errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(target_urls))) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in target_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    img = future.result()
                    if img: pil_images.append(img)
                    else: download_errors.append(f"下载失败: {url}")
                except Exception as e:
                    download_errors.append(f"下载异常: {e}")
        
        return pil_images, download_errors


# --- 主要节点: GrsaiGptImage ---
class GrsaiGptImage:
    CATEGORY = "Nkxx/Grsai/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat running on the grass"}),
                "size": (GPT_IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1, "label": "并发任务数"}),
                "model": (GPT_IMAGE_MODEL_CHOICES, {"default": "gpt-image-2"}),
                "resolution": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "image_5": ("IMAGE",), "image_6": ("IMAGE",), "image_7": ("IMAGE",), "image_8": ("IMAGE",),
                "image_9": ("IMAGE",), "image_10": ("IMAGE",), "image_11": ("IMAGE",), "image_12": ("IMAGE",),
                "image_13": ("IMAGE",), "image_14": ("IMAGE",), "image_15": ("IMAGE",), "image_16": ("IMAGE",),
                "quality": (["", "auto", "low", "medium", "high"], {"default": ""}),
            }
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (GPT_IMAGE_MODEL_CHOICES, {"default": "gpt-image-2"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat running on the grass"}),
                "aspectRatio": (GPT_IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "imageSize": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
            },
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "image_5": ("IMAGE",), "image_6": ("IMAGE",), "image_7": ("IMAGE",), "image_8": ("IMAGE",),
                "image_9": ("IMAGE",), "image_10": ("IMAGE",), "image_11": ("IMAGE",), "image_12": ("IMAGE",),
                "image_13": ("IMAGE",), "image_14": ("IMAGE",), "image_15": ("IMAGE",), "image_16": ("IMAGE",),
                "quality": (["", "auto", "low", "medium", "high"], {"default": ""}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1, "label": "concurrency"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status")
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _get_credits_balance(self, api_key: str) -> str:
        try:
            url = f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 0: return str(int(data["data"]["credits"]))
        except: pass
        return "N/A"

    def execute(
        self,
        prompt: str = "",
        size: str = None,
        concurrency: int = 1,
        model: str = "gpt-image-2",
        resolution: str = "默认",
        api_key: str = "",
        quality: str = "",
        aspectRatio: str = "1:1",
        imageSize: str = "默认",
        **kwargs
    ):
        if size is not None:
            aspectRatio = size
        if resolution is not None and size is not None:
            imageSize = resolution
        final_api_key = resolve_api_key(api_key, "grsai")
        if not final_api_key: 
            return (torch.zeros((1, 1, 1, 3)), "错误: 未配置 Grsai API Key。")

        # 1. 提取并上传参考图 (调用 utils.py)
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 17) if kwargs.get(f"image_{i}") is not None]
        uploaded_urls = []
        for img_tensor in images_in:
            url = upload_image_grsai(final_api_key, img_tensor)
            if url: uploaded_urls.append(url)
            else: return (torch.zeros((1, 1, 1, 3)), "错误: 部分参考图上传失败。")

        # 2. 并发执行生成任务
        try:
            api_client = GrsaiGPTAPI(api_key=final_api_key)
            all_pil_images, all_errors = [], []

            def task_runner(_):
                try:
                    # 将 variants 参数固定写死为 1
                    return api_client.gpt_image_generate(prompt, uploaded_urls, model, aspectRatio, imageSize, quality, 1)
                except Exception as e:
                    return [], [str(e)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(task_runner, range(concurrency))
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: all_errors.extend(errs)

            # 3. 结果处理
            if not all_pil_images:
                err_msg = "; ".join(all_errors) if all_errors else "未知错误"
                return (torch.zeros((1, 1, 1, 3)), f"生成失败: {err_msg}")

            final_tensor = pil_to_tensor(all_pil_images)
            credits = self._get_credits_balance(final_api_key)
            status_text = f"成功: {len(all_pil_images)} 张 (任务 x{concurrency}, 变体 x1) | 失败: {len(all_errors)} | 积分: {credits}"
            
            return {"ui": {"string": [status_text]}, "result": (final_tensor, status_text)}

        except Exception as e:
            traceback.print_exc()
            return (torch.zeros((1, 1, 1, 3)), f"系统错误: {str(e)}")

# --- 节点注册 ---
gpt_image_task_db = TaskManager("grsai_gpt_image_async_history.json", max_completed_history=15)


def _get_next_gpt_task_number(tasks: dict, is_batch: bool = False) -> int:
    prefix = "GPT批量任务" if is_batch else "GPT任务"
    nums = []
    for task_id in tasks.keys():
        if task_id.startswith(prefix):
            if not is_batch and task_id.startswith("GPT批量任务"):
                continue
            try:
                nums.append(int(task_id[len(prefix):]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


class GrsaiGptImageAsyncAPI:
    def __init__(self, api_key: str):
        self.base_url = "https://grsai.dakka.com.cn"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def post(self, endpoint: str, payload: dict, timeout: int = 120) -> dict:
        response = requests.post(f"{self.base_url}{endpoint}", headers=self.headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("code") and data.get("code") != 0:
            raise GrsaiAPIError(f"API error: {data.get('msg', 'unknown')}")
        return data


class _GrsaiGptImageAsyncBase:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error_text(self, message: str):
        return {"ui": {"string": [message]}, "result": (message,)}

    def _error_image(self, message: str):
        return {"ui": {"string": [message]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), message)}

    def _get_credits_balance(self, api_key: str) -> str:
        try:
            url = f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 0 and data.get("data", {}).get("credits") is not None:
                    return str(int(data["data"]["credits"]))
        except Exception:
            pass
        return "N/A"

    def _upload_images(self, api_key: str, images_in: list) -> list[str]:
        urls = []
        for image in images_in:
            if image is None:
                continue
            url = upload_image_grsai(api_key, image)
            if not url:
                raise GrsaiAPIError("参考图上传失败")
            urls.append(url)
        return urls

    def _build_async_payload(self, api_key: str, prompt: str, urls: list[str], model: str, aspect_ratio: str, image_size: str, quality: str) -> dict:
        payload = GrsaiGPTAPI(api_key)._build_gpt_image_payload(
            prompt=prompt,
            urls=urls,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=image_size,
            quality=quality,
            variants=1,
        )
        payload["webHook"] = "-1"
        payload["shutProgress"] = True
        return payload

    def _extract_task_id(self, response: dict) -> str | None:
        if not isinstance(response, dict):
            return None
        return response.get("id") or response.get("data", {}).get("id")


class GrsaiGptImageAsyncSubmit(_GrsaiGptImageAsyncBase):
    CATEGORY = "Nkxx/Grsai/GPT Image Async"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (GPT_IMAGE_MODEL_CHOICES, {"default": "gpt-image-2"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat running on the grass"}),
                "aspectRatio": (GPT_IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "imageSize": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
            },
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "image_5": ("IMAGE",), "image_6": ("IMAGE",), "image_7": ("IMAGE",), "image_8": ("IMAGE",),
                "image_9": ("IMAGE",), "image_10": ("IMAGE",), "image_11": ("IMAGE",), "image_12": ("IMAGE",),
                "image_13": ("IMAGE",), "image_14": ("IMAGE",), "image_15": ("IMAGE",), "image_16": ("IMAGE",),
                "quality": (["", "auto", "low", "medium", "high"], {"default": ""}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "label": "concurrency"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
            },
        }

    def submit(self, model: str, prompt: str, aspectRatio: str, imageSize: str, quality: str = "", concurrency: int = 1, api_key: str = "", **kwargs):
        final_api_key = resolve_api_key(api_key, "grsai")
        if not final_api_key:
            return self._error_text("未配置 Grsai API Key。")

        try:
            images_in = [kwargs.get(f"image_{i}") for i in range(1, 17) if kwargs.get(f"image_{i}") is not None]
            uploaded_urls = self._upload_images(final_api_key, images_in)
            api = GrsaiGptImageAsyncAPI(final_api_key)
            subtasks = []

            for _ in range(concurrency):
                payload = self._build_async_payload(final_api_key, prompt, uploaded_urls, model, aspectRatio, imageSize, quality)
                response = api.post("/v1/draw/completions", payload)
                task_id = self._extract_task_id(response)
                if not task_id:
                    raise GrsaiAPIError(f"提交成功但未返回任务 id: {response}")
                subtasks.append({"api_task_id": task_id, "status": "running", "image_url": None, "progress": 0, "failure_reason": None})

            with gpt_image_task_db.lock:
                tasks = gpt_image_task_db.read_tasks()
                local_task_id = f"GPT任务{_get_next_gpt_task_number(tasks)}"

            gpt_image_task_db.update_task(local_task_id, {
                "type": "normal",
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspectRatio,
                "image_size": imageSize,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "concurrency": concurrency,
                "subtasks": subtasks,
            })

            status = f"GPT Image 异步提交成功 | {local_task_id} | 子任务: {len(subtasks)} | 积分: {self._get_credits_balance(final_api_key)}"
            return {"ui": {"string": [status]}, "result": (status,)}
        except Exception as e:
            traceback.print_exc()
            return self._error_text(f"提交失败: {e}")


class GrsaiGptImageAsyncBatchSubmit(_GrsaiGptImageAsyncBase):
    CATEGORY = "Nkxx/Grsai/GPT Image Async"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "CSV/Excel 文件路径"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "model": (GPT_IMAGE_MODEL_CHOICES, {"default": "gpt-image-2"}),
                "aspectRatio": (GPT_IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "imageSize": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
                "quality": (["", "auto", "low", "medium", "high"], {"default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
            },
        }

    def submit_batch(self, file_path: str, column_name: str, prompt_prefix: str, model: str, aspectRatio: str, imageSize: str, executions_per_prompt: int, images=None, quality: str = "", api_key: str = ""):
        final_api_key = resolve_api_key(api_key, "grsai")
        if not final_api_key:
            return self._error_text("未配置 Grsai API Key。")
        if not file_path or not os.path.exists(file_path):
            return self._error_text("文件路径为空或文件不存在。")

        try:
            if file_path.lower().endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")
            elif file_path.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            else:
                return self._error_text("仅支持 .csv/.xls/.xlsx 文件。")
            if column_name not in df.columns:
                return self._error_text(f"列不存在: {column_name}")

            base_prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()]
            prompts = [p for p in base_prompts for _ in range(max(1, executions_per_prompt))]
            if not prompts:
                return self._error_text("未读取到有效 prompt。")

            images_in = []
            if images is not None:
                if isinstance(images, torch.Tensor) and len(images.shape) == 4:
                    images_in = [images[i].unsqueeze(0) for i in range(images.shape[0])]
                elif isinstance(images, torch.Tensor):
                    images_in = [images]
                elif isinstance(images, list):
                    images_in = images
            uploaded_urls = self._upload_images(final_api_key, images_in)

            api = GrsaiGptImageAsyncAPI(final_api_key)
            subtasks, errors = [], []

            def submit_one(idx: int, prompt_text: str):
                try:
                    payload = self._build_async_payload(final_api_key, prompt_text, uploaded_urls, model, aspectRatio, imageSize, quality)
                    response = api.post("/v1/draw/completions", payload)
                    task_id = self._extract_task_id(response)
                    if not task_id:
                        raise GrsaiAPIError(f"未返回任务 id: {response}")
                    return {"api_task_id": task_id, "status": "running", "image_url": None, "progress": 0, "failure_reason": None, "original_prompt": prompt_text}
                except Exception as exc:
                    return {"error": str(exc), "index": idx}

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(submit_one, i, p) for i, p in enumerate(prompts)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if "error" in result:
                        errors.append(result["error"])
                    else:
                        subtasks.append(result)

            if not subtasks:
                return self._error_text(f"批量提交全部失败: {errors[0] if errors else 'unknown'}")

            with gpt_image_task_db.lock:
                tasks = gpt_image_task_db.read_tasks()
                local_task_id = f"GPT批量任务{_get_next_gpt_task_number(tasks, is_batch=True)}"

            gpt_image_task_db.update_task(local_task_id, {
                "type": "batch",
                "prompt": f"批量文件: {os.path.basename(file_path)}",
                "model": model,
                "aspect_ratio": aspectRatio,
                "image_size": imageSize,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "concurrency": len(subtasks),
                "subtasks": subtasks,
            })

            status = f"GPT Image 批量提交完成 | {local_task_id} | 成功: {len(subtasks)}/{len(prompts)} | 失败: {len(errors)} | 积分: {self._get_credits_balance(final_api_key)}"
            return {"ui": {"string": [status]}, "result": (status,)}
        except Exception as e:
            traceback.print_exc()
            return self._error_text(f"批量提交失败: {e}")


class GrsaiGptImageAsyncQuery(_GrsaiGptImageAsyncBase):
    CATEGORY = "Nkxx/Grsai/GPT Image Async"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status")
    FUNCTION = "query_and_download"

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"})}}

    def _status_text(self, tasks: dict) -> str:
        lines = []
        status_map = {"running": "运行中", "succeeded": "待下载", "failed": "失败", "downloaded": "已下载"}
        for task_id, info in sorted(tasks.items(), key=lambda x: x[1].get("submitted_at", ""), reverse=True):
            subs = info.get("subtasks", [])
            ok = sum(1 for sub in subs if sub.get("status") == "succeeded")
            fail = sum(1 for sub in subs if sub.get("status") == "failed")
            status = status_map.get(info.get("status", "running"), info.get("status", "running"))
            lines.append(f"[{status}] {task_id} | {ok}/{len(subs)} 成功 | {fail} 失败 | {info.get('model', '')}")
        return "\n".join(lines) if lines else "当前没有 GPT Image 异步任务。"

    def query_and_download(self, api_key: str = ""):
        final_api_key = resolve_api_key(api_key, "grsai")
        if not final_api_key:
            return self._error_image("未配置 Grsai API Key。")

        tasks = gpt_image_task_db.read_tasks()
        if not tasks:
            return self._error_image("当前没有 GPT Image 异步任务。")

        api = GrsaiGptImageAsyncAPI(final_api_key)
        updated = False
        for task_id, info in list(tasks.items()):
            if info.get("status") in ("succeeded", "failed", "downloaded"):
                continue
            subtasks = info.get("subtasks", [])
            any_running = False
            any_succeeded = False
            sub_updated = False
            for sub in subtasks:
                if sub.get("status") in ("succeeded", "failed"):
                    any_succeeded = any_succeeded or sub.get("status") == "succeeded"
                    continue
                status = "running"
                try:
                    response = api.post("/v1/draw/result", {"id": sub.get("api_task_id")}, timeout=60)
                    data = response.get("data", response)
                    status = data.get("status", "running")
                    if status == "pending":
                        status = "running"
                    sub["progress"] = data.get("progress", sub.get("progress", 0))
                    if data.get("results"):
                        sub["image_url"] = data["results"][0].get("url")
                    elif data.get("url"):
                        sub["image_url"] = data.get("url")
                    if status == "failed":
                        sub["failure_reason"] = data.get("failure_reason") or data.get("error") or "unknown"
                    if sub.get("status") != status:
                        sub["status"] = status
                        sub_updated = True
                except Exception:
                    status = "running"
                any_running = any_running or status == "running"
                any_succeeded = any_succeeded or status == "succeeded"

            final_status = "running" if any_running else ("succeeded" if any_succeeded else "failed")
            if final_status != info.get("status") or sub_updated:
                info["status"] = final_status
                info["subtasks"] = subtasks
                gpt_image_task_db.update_task(task_id, info)
                updated = True

        if updated:
            tasks = gpt_image_task_db.read_tasks()

        status_text = self._status_text(tasks)
        ready = [(task_id, info) for task_id, info in tasks.items() if info.get("status") == "succeeded"]
        if not ready:
            return {"ui": {"string": [status_text]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), status_text)}

        ready.sort(key=lambda x: x[1].get("submitted_at", "2999-01-01"))
        target_id, target_info = ready[0]
        pil_images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(download_image, sub.get("image_url"))
                for sub in target_info.get("subtasks", [])
                if sub.get("status") == "succeeded" and sub.get("image_url")
            ]
            for future in concurrent.futures.as_completed(futures):
                image = future.result()
                if image:
                    pil_images.append(image)

        if not pil_images:
            return self._error_image(f"{target_id} 图片下载失败。")

        gpt_image_task_db.update_task(target_id, {"status": "downloaded"})
        tasks = gpt_image_task_db.read_tasks()
        final_text = f"下载成功: {target_id} | 图片: {len(pil_images)} | 积分: {self._get_credits_balance(final_api_key)}\n{self._status_text(tasks)}"
        return {"ui": {"string": [final_text]}, "result": (safe_pil_batch_to_tensor(pil_images), final_text)}


NODE_CLASS_MAPPINGS = {
    "GrsaiGptImage": GrsaiGptImage,
    "GrsaiGptImageAsyncSubmit": GrsaiGptImageAsyncSubmit,
    "GrsaiGptImageAsyncBatchSubmit": GrsaiGptImageAsyncBatchSubmit,
    "GrsaiGptImageAsyncQuery": GrsaiGptImageAsyncQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiGptImage": "Grsai GPT Image",
    "GrsaiGptImageAsyncSubmit": "Grsai GPT Image 异步提交",
    "GrsaiGptImageAsyncBatchSubmit": "Grsai GPT Image 异步批量提交",
    "GrsaiGptImageAsyncQuery": "Grsai GPT Image 异步查询下载",
}
