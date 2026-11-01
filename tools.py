# tools.py
import os
import time
import json
import requests
import torch
import folder_paths
from server import PromptServer
from aiohttp import web
from .utils import tensor_to_pil

# ======================================================================
# 模型热更新配置中心
# ======================================================================
# 【核心修改】：缓存文件现在永久存放在当前节点目录下的 data 文件夹内
MODELS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "wujiai_local_models_cache.json")

# 指向你的 Gitee 仓库 Raw 链接
GITEE_MODELS_RAW_URL = "https://gitee.com/jiege777/app-config/raw/master/models.json"

# 终极兜底配置
DEFAULT_MODELS_CONFIG = {
    "顶部公告": [
        "欢迎使用我的工作流，关注B站那颗星星188！",
    ],
    "图像模型": [
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "doubao-seedream-5-0-260128",
        "doubao-seedream-4-5-251128",
        "doubao-seedream-4-0-250828",
        "gpt-image-1.5-all",
        "grok-4.2-image",
        "grok-4.1-image",
        "grok-4-image"
    ],
    "视频模型": [
        "grok-video-3",
        "grok-video-3-10s",
        "grok-video-3-15s",
        "sora-2",
        "veo_3_1-fast",
        "veo_3_1-fast-4K",
        "veo_3_1-components",
        "veo_3_1-components-4K"
    ],
    "大语言模型": [
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gpt-5.4",
        "gpt-5.3-chat-latest",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5-mini-2025-08-07",
        "grok-4.1",
        "grok-4",
        "grok-4-fast",
        "doubao-seed-1-8-251228"
    ]
}

# 增加两个全局变量，用于内存缓存和频率控制
_LAST_FETCH_TIME = 0
_MEMORY_MODELS_CACHE = None

def fetch_models_config():
    """拉取或读取本地模型配置 (带防封控与内存缓存机制)"""
    global _LAST_FETCH_TIME, _MEMORY_MODELS_CACHE
    config = DEFAULT_MODELS_CONFIG
    
    # 1. 频率限制：如果距离上次拉取不到 10 秒，直接返回内存数据
    current_time = time.time()
    if _MEMORY_MODELS_CACHE and (current_time - _LAST_FETCH_TIME < 10):
        return _MEMORY_MODELS_CACHE
    
    # 2. 尝试从 Gitee 拉取 
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(GITEE_MODELS_RAW_URL, headers=headers, timeout=5)
        
        if response.status_code == 200:
            raw_text = response.text
            raw_text = raw_text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
            
            fetched_data = json.loads(raw_text)
            
            if isinstance(fetched_data, dict) and len(fetched_data) > 0:
                config = fetched_data
                
                _MEMORY_MODELS_CACHE = config
                _LAST_FETCH_TIME = current_time
                
                # 【新增】：写入前确保 data 文件夹存在，防止报错
                os.makedirs(os.path.dirname(MODELS_CACHE_FILE), exist_ok=True)
                
                with open(MODELS_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                return config
        else:
            print(f"[Wujiai Tools] Gitee 拒绝请求，状态码: {response.status_code}")
            
    except Exception as e:
        print(f"[Wujiai Tools] 从 Gitee 获取模型列表失败: {e}")

    # 3. 兜底：网络失败时，读取本地文件缓存
    if os.path.exists(MODELS_CACHE_FILE):
        try:
            with open(MODELS_CACHE_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                _MEMORY_MODELS_CACHE = config 
        except Exception as e:
            print(f"[Wujiai Tools] 读取本地缓存失败，使用硬编码默认配置。({e})")
            
    return config

@PromptServer.instance.routes.get("/nkxx/wujiai/models_config")
async def get_models_config(request):
    config = fetch_models_config()
    return web.json_response(config)

# ======================================================================
# 模型独立展示板 (无连线纯视觉节点)
# ======================================================================
class WujiaiModelBrowser:
    CATEGORY = "Nkxx/Tools"
    
    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    OUTPUT_NODE = True 

    @classmethod
    def INPUT_TYPES(cls):
        config = fetch_models_config()
        categories = [k for k in config.keys() if k != "顶部公告"] if config else ["未获取到分类"]

        return {
            "required": {
                "model_list_display": ("STRING", {"multiline": True, "default": "正在加载最新模型列表..."}),
                "category": (categories, {"default": categories[0] if categories else ""}),
            }
        }

    def do_nothing(self, **kwargs):
        return {}


# ======================================================================
# API Key 集中管理器 (孤立记事本纯净版)
# ======================================================================
class ApiKeyManager:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "active_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
            },
            "optional": {}
        }
        
        for i in range(1, 11):
            inputs["optional"][f"name{i}"] = ("STRING", {"default": f""})
            inputs["optional"][f"api_key{i}"] = ("STRING", {"default": ""})
            
        return inputs

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "Nkxx/Tools"
    
    # 设为 True 保证不连线也能被保存到工作流中
    OUTPUT_NODE = True 

    def do_nothing(self, **kwargs):
        return {}


# ======================================================================
# 原有的 Tools 节点
# ======================================================================
class GroupRunnerConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "Nkxx/Tools"
    OUTPUT_NODE = True 

    def do_nothing(self):
        return {}


class DynamicImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像列表 (IMAGE)",)
    FUNCTION = "combine_to_list"
    CATEGORY = "Nkxx/Tools"

    def combine_to_list(self, inputcount, **kwargs):
        images = []
        for i in range(1, inputcount + 1):
            key = f"image_{i}"
            if key in kwargs:
                incoming = kwargs[key]
                if incoming is None:
                    continue
                if isinstance(incoming, list):
                    images.extend(incoming)
                else:
                    images.append(incoming)
        if len(images) == 0:
            return ([],)  
        return (images,)


class SaveImageByFilename:
    CATEGORY = "Nkxx/Tools"
    OUTPUT_NODE = True
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"multiline": True}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()

    def execute(self, images, filenames, output_path="", suffix=""):
        pil_images = tensor_to_pil(images)
        paths = [p.strip() for p in filenames.strip().split('\n') if p.strip()]

        if not paths:
            print("Warning: SaveImageByFilename 节点收到空的 'filenames' 输入。")
            return {"ui": {"string": ["错误: 文件名列表为空"]}}
            
        if len(pil_images) != len(paths):
            print(f"Warning: 图像数量 ({len(pil_images)}) 与文件名数量 ({len(paths)}) 不匹配。将只保存较少数量的图像。")

        main_output_dir = folder_paths.get_output_directory()
        saved_count = 0
        ui_image_info = []

        for i, pil_image in enumerate(pil_images):
            if i >= len(paths): break 

            original_path = paths[i]
            
            if output_path:
                final_save_dir = os.path.join(main_output_dir, output_path)
                original_filename = os.path.basename(original_path)
                base, ext = os.path.splitext(original_filename)
            else:
                final_save_dir = os.path.dirname(original_path)
                original_filename = os.path.basename(original_path)
                base, ext = os.path.splitext(original_filename)
            
            if not ext.lower() == ".png": ext = ".png"

            os.makedirs(final_save_dir, exist_ok=True)
            
            new_filename_no_ext = f"{base}{suffix}"
            new_path = os.path.join(final_save_dir, new_filename_no_ext + ext)
            
            counter = 1
            while os.path.exists(new_path):
                new_path = os.path.join(final_save_dir, f"{new_filename_no_ext} ({counter}){ext}")
                counter += 1

            pil_image.save(new_path, "PNG", compress_level=4)
            saved_count += 1
            
            ui_filename = os.path.relpath(new_path, main_output_dir)
            ui_subfolder = os.path.dirname(ui_filename)
            ui_image_info.append({"filename": os.path.basename(ui_filename), "subfolder": ui_subfolder, "type": "output"})

        return {"ui": {"images": ui_image_info, "string": [f"成功保存 {saved_count} 张高清图"]}}

# ======================================================================
# 统一节点注册
# ======================================================================
NODE_CLASS_MAPPINGS = {
    "GroupRunnerConsole": GroupRunnerConsole,
    "DynamicImageList": DynamicImageList,
    "SaveImageByFilename": SaveImageByFilename,
    "WujiaiModelBrowser": WujiaiModelBrowser,
    "ApiKeyManager": ApiKeyManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroupRunnerConsole": "快速运行中控台",
    "DynamicImageList": "图像动态列表",
    "SaveImageByFilename": "💾 智能按文件名保存",
    "WujiaiModelBrowser": "可选模型面板",
    "ApiKeyManager": "🔑 API Key 管理器"
}