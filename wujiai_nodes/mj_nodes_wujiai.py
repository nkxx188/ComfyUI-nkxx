# wujiai_nodes/mj_nodes_wujiai.py
import os
import requests
import time
import json
import re
import traceback
import torch
import numpy as np

import server
from aiohttp import web

# 从公共工具箱导入核心底座
from ..utils import (
    TaskManager,
    upload_image_wujiai,
    robust_download_video,
    VideoAdapter,
    get_wujiai_host,
    download_image,              
    safe_pil_batch_to_tensor
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
mj_task_db = TaskManager("wujiai_mj_history.json", max_completed_history=20)

# ======================================================================
# 为前端 JS 按钮提供的专属数据接口
# ======================================================================
@server.PromptServer.instance.routes.get("/wujiai_mj/get_recent_tasks")
async def get_recent_tasks(request):
    """供前端调用的 API，返回最近带有按钮的有效任务列表"""
    with mj_task_db.lock:
        tasks = mj_task_db.read_tasks()
        
    recent_tasks = ["--- 请选择近期任务 ---"]
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True)
    
    for tid, info in sorted_tasks:
        if "buttons" in info and len(info["buttons"]) > 0:
            prompt_snippet = info.get("prompt", "")[:20].replace('\n', ' ')
            recent_tasks.append(f"{tid} | {prompt_snippet}...")
            
    if len(recent_tasks) == 1:
        recent_tasks.append("暂无包含按钮的历史任务")
        
    return web.json_response(recent_tasks)


# ======================================================================
# 节点 1：MJ Imagine (生图提交)
# ======================================================================
class MJImagineSubmit:
    CATEGORY = "Nkxx/Wujiai/MJ"
    OUTPUT_NODE = True 
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cyberpunk cat"}),
                "model_version": (["v7", "v6.1", "niji 7", "niji 6", "v6.0", "v5.2"], {"default": "v7"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "3:4", "4:3", "2:3", "3:2"], {"default": "1:1"}),
            },
            "optional": {
                "extra_args": ("STRING", {"default": "", "multiline": False, "tooltip": "可选附加参数，例如: --iw 1.5 --cw 50"}),
                "api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"}),
                "image_prompts": ("IMAGE", {"tooltip": "常规垫图 (图生图)"}),
                "sref_images": ("IMAGE", {"tooltip": "风格参考 (--sref)"}),
                "character_refs": ("IMAGE", {"tooltip": "角色/物体参考 (自动适配 --cref / --oref)"}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit(self, prompt, model_version, aspect_ratio, extra_args="", api_key="", 
               image_prompts=None, sref_images=None, character_refs=None):
        
        final_api_key = resolve_api_key(api_key, "wujiai")
        if not final_api_key: 
            return {"ui": {"string": ["请配置 Wujiai API Key"]}, "result": ("请配置 Wujiai API Key",)}

        try:
            # === 核心修复点：增加了对 list 类型的判断与遍历兼容 ===
            def tensor_to_urls(image_input):
                if image_input is None: return []
                
                # 判断传入的是列表 (DynamicImageList) 还是单一 Tensor，统一转为列表处理
                image_list = image_input if isinstance(image_input, list) else [image_input]
                urls = []
                
                for img_data in image_list:
                    # 遍历每一个 tensor 的 batch 维度
                    for i in range(img_data.shape[0]):
                        img_tensor = img_data[i:i+1]
                        url = upload_image_wujiai(final_api_key, img_tensor)
                        if url: urls.append(url)
                return urls
            # ====================================================

            img2img_urls = tensor_to_urls(image_prompts)
            sref_urls = tensor_to_urls(sref_images)
            character_urls = tensor_to_urls(character_refs)

            final_prompt = prompt.strip()
            display_prompt = prompt.strip()
            
            # 1. 常规垫图 (拼接在最前)
            if img2img_urls:
                final_prompt = " ".join(img2img_urls) + " " + final_prompt
                img_disp = " ".join([f"[img2img_url_{i+1}]" for i in range(len(img2img_urls))])
                display_prompt = img_disp + " " + display_prompt

            # 2. 风格参考 SREF
            if sref_urls:
                sref_str = " ".join(sref_urls)
                sref_disp = " ".join([f"[sref_url_{i+1}]" for i in range(len(sref_urls))])
                if "--sref" in final_prompt:
                    final_prompt = final_prompt.replace("--sref", f"--sref {sref_str}", 1)
                    display_prompt = display_prompt.replace("--sref", f"--sref {sref_disp}", 1)
                else:
                    final_prompt += f" --sref {sref_str}"
                    display_prompt += f" --sref {sref_disp}"

            # 3. 角色参考 CREF / OREF
            if character_urls:
                char_str = " ".join(character_urls)
                char_disp = " ".join([f"[cref_url_{i+1}]" for i in range(len(character_urls))])
                ref_tag = "--oref" if "7" in model_version else "--cref"
                
                if ref_tag in final_prompt:
                    final_prompt = final_prompt.replace(ref_tag, f"{ref_tag} {char_str}", 1)
                    display_prompt = display_prompt.replace(ref_tag, f"{ref_tag} {char_disp}", 1)
                else:
                    final_prompt += f" {ref_tag} {char_str}"
                    display_prompt += f" {ref_tag} {char_disp}"

            # 4. 宽高比
            if not re.search(r'--(ar|aspect)\b', final_prompt):
                final_prompt += f" --ar {aspect_ratio}"
                display_prompt += f" --ar {aspect_ratio}"

            # 5. 附加参数
            if extra_args and extra_args.strip():
                final_prompt += f" {extra_args.strip()}"
                display_prompt += f" {extra_args.strip()}"

            # 6. 版本号
            bot_type = "NIJI_JOURNEY" if "niji" in model_version else "MID_JOURNEY"
            v_suffix = f"--{model_version}" if "niji" in model_version else f"--v {model_version.replace('v', '')}"
            if not re.search(rf'--(v|niji)\b', final_prompt):
                final_prompt += f" {v_suffix}"
                display_prompt += f" {v_suffix}"

            proxy_model = "mj_imagine"
            payload = {
                "botType": bot_type,
                "prompt": final_prompt.strip(),
                "base64Array": [],
                "model": proxy_model
            }
            
            headers = {"Authorization": f"Bearer {final_api_key}", "Content-Type": "application/json"}
            
            print("\n" + "="*50)
            print("[MJ Debug] 正在向 API 提交 Imagine 任务...")
            print(json.dumps(payload, indent=4, ensure_ascii=False))
            print("="*50 + "\n")

            res = requests.post(f"{HOST}/mj/submit/imagine", headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            
            task_id = str(data.get("result", ""))
            if not task_id or data.get("code") != 1:
                raise Exception(f"提交失败: {data}")

            # 调用 TaskManager 更新数据库
            mj_task_db.update_task(task_id, {
                "type": "IMAGINE",
                "model": proxy_model,
                "prompt": display_prompt, 
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running"
            })

            msg = f"MJ {proxy_model} 提交成功!\nTask ID: {task_id}\nPrompt:\n{display_prompt}"
            return {"ui": {"string": [msg]}, "result": (msg,)}
            
        except Exception as e:
            traceback.print_exc()
            err = f"提交失败: {e}"
            return {"ui": {"string": [err]}, "result": (err,)}


# ======================================================================
# 节点 2：MJ 全局查询
# ======================================================================
class MJQueryTasks:
    CATEGORY = "Nkxx/Wujiai/MJ"
    OUTPUT_NODE = True
    
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
        if not final_api_key: 
            return {"ui": {"string": ["请配置 API Key"]}, "result": ("请配置 API Key",)}

        with mj_task_db.lock:
            tasks = mj_task_db.read_tasks()
            
        headers = {"Authorization": f"Bearer {final_api_key}"}
        pending_tasks = {k: v for k, v in tasks.items() if v.get("status") in ["running", "SUBMITTED", "IN_PROGRESS"]}
        
        for tid, tinfo in pending_tasks.items():
            try:
                res = requests.get(f"{HOST}/mj/task/{tid}/fetch", headers=headers, timeout=15)
                if res.status_code == 200:
                    d = res.json()
                    new_status = d.get("status", "running")
                    
                    if new_status == "SUCCESS":
                        mj_task_db.update_task(tid, {
                            "status": "SUCCESS", 
                            "image_url": d.get("imageUrl"),
                            "buttons": d.get("buttons", [])
                        })
                    elif new_status == "FAILURE":
                        mj_task_db.update_task(tid, {
                            "status": "FAILURE", 
                            "failReason": d.get("failReason", "未知错误")
                        })
            except Exception:
                pass
                
        final_tasks = mj_task_db.read_tasks()
        report = ["--- MJ 任务队列 ---"]
        
        for tid, info in sorted(final_tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
            status = info.get("status", "unknown")
            pt = info.get("prompt", "")[:30] + "..."
            display_model = info.get("model", info.get("type", "UNKNOWN"))
            report.append(f"[{status}] {display_model} | {tid} | {pt}")
            
        full_msg = "\n".join(report)
        return {"ui": {"string": [full_msg]}, "result": (full_msg,)}


# ======================================================================
# 节点 3：MJ 动作执行 (U/V/Reroll)
# ======================================================================
class MJActionSubmit:
    CATEGORY = "Nkxx/Wujiai/MJ"
    OUTPUT_NODE = True 
    
    @classmethod
    def INPUT_TYPES(cls):
        # 这里的下拉选项仅作初始化占位，真实的选项由前端 JS 动态刷新注入
        with mj_task_db.lock:
            tasks = mj_task_db.read_tasks()
            
        recent_tasks = ["--- 请选择近期任务 ---"]
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True)
        
        for tid, info in sorted_tasks:
            if "buttons" in info and len(info["buttons"]) > 0:
                prompt_snippet = info.get("prompt", "")[:20].replace('\n', ' ')
                recent_tasks.append(f"{tid} | {prompt_snippet}...")
                
        if len(recent_tasks) == 1:
            recent_tasks.append("暂无包含按钮的历史任务")

        return {
            "required": {
                "recent_task": (recent_tasks,),
                # [恢复保留 🔄 符号]
                "action": (["U1", "U2", "U3", "U4", "V1", "V2", "V3", "V4", "🔄 Reroll"], {"default": "U1"}),
            },
            "optional": {
                "variation_mode": (["默认", "高变化", "低变化"], {"default": "默认"}),
                "task_id_override": ("STRING", {"default": "", "tooltip": "优先使用此ID覆盖上方选择"}),
                "api_key": ("STRING", {"default": ""})
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, recent_task, action, variation_mode="默认", task_id_override="", api_key=""):
        return True
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "submit_action"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit_action(self, recent_task, action, variation_mode="默认", task_id_override="", api_key=""):
        final_api_key = resolve_api_key(api_key, "wujiai")
        
        target_task_id = ""
        if task_id_override and task_id_override.strip():
            target_task_id = task_id_override.strip()
        elif recent_task and recent_task != "--- 请选择近期任务 ---" and recent_task != "暂无包含按钮的历史任务":
            target_task_id = recent_task.split(" | ")[0].strip()
            
        if not final_api_key or not target_task_id: 
            msg = "错误: 缺少 API_KEY 或未选择/填写 TASK_ID"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        # 内部映射: 匹配 Discord API 时 Reroll 必须要用到 🔄 这个 emoji
        target_label = "🔄" if "Reroll" in action else action

        if action.startswith("U"): proxy_model = "mj_upscale"
        elif action.startswith("V"): 
            if "低变化" in variation_mode: proxy_model = "mj_low_variation"
            elif "高变化" in variation_mode: proxy_model = "mj_high_variation"
            else: proxy_model = "mj_variation"
        elif "Reroll" in action: proxy_model = "mj_reroll"
        else: proxy_model = "mj_action"

        with mj_task_db.lock:
            tasks = mj_task_db.read_tasks()
            
        task_info = tasks.get(target_task_id)
        
        if not task_info or "buttons" not in task_info:
            msg = f"找不到任务 {target_task_id} 的按钮数据，请确保 Query 节点刷新过它。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        custom_id = None
        for btn in task_info["buttons"]:
            if btn.get("label") == target_label or btn.get("emoji") == target_label:
                custom_id = btn.get("customId")
                break
                
        if not custom_id:
            msg = f"任务 {target_task_id} 中找不到 {action} 按钮。它可能不是四宫格任务。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        try:
            payload = {
                "customId": custom_id,
                "taskId": target_task_id,
                "chooseSameChannel": True,
                "model": proxy_model
            }
            headers = {"Authorization": f"Bearer {final_api_key}", "Content-Type": "application/json"}
            
            print("\n" + "="*50)
            print(f"[MJ Debug] 正在向 API 提交 Action ({action}) 任务...")
            print(json.dumps(payload, indent=4, ensure_ascii=False))
            print("="*50 + "\n")

            res = requests.post(f"{HOST}/mj/submit/action", headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()

            new_task_id = str(data.get("result", ""))
            if not new_task_id or data.get("code") != 1:
                raise Exception(f"API 拒绝请求: {data}")

            mj_task_db.update_task(new_task_id, {
                "type": f"ACTION_{action}",
                "model": proxy_model,
                "prompt": f"Based on {target_task_id}",
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running"
            })

            success_msg = f"[{proxy_model}] 提交成功!\n动作: {action}\n旧任务: {target_task_id}\n新任务 ID: {new_task_id}\n请随后点击查询节点刷新状态。"
            return {"ui": {"string": [success_msg]}, "result": (success_msg,)}

        except Exception as e:
            traceback.print_exc()
            err_msg = f"Action 异常: {e}"
            return {"ui": {"string": [err_msg]}, "result": (err_msg,)}


# ======================================================================
# 节点 4：MJ 获取结果 (图片下载与智能切分)
# ======================================================================
class MJGetResult:
    CATEGORY = "Nkxx/Wujiai/MJ"
    
    @classmethod
    def INPUT_TYPES(cls): 
        return {
            "optional": {
                "specific_task_id": ("STRING", {"default": "", "tooltip": "留空则自动拿最老的一个未下载成功任务"}),
                "split_grid": (["Yes (自动切分四宫格)", "No (保持原图)"], {"default": "Yes (自动切分四宫格)"})
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "get_result"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_result(self, specific_task_id="", split_grid="Yes (自动切分四宫格)"):
        # 生成一个黑色的空图片占位符，防止前端弹窗报错
        empty_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        with mj_task_db.lock:
            tasks = mj_task_db.read_tasks()
            
        target_id = None
        
        if specific_task_id.strip():
            target_id = specific_task_id.strip()
            if tasks.get(target_id, {}).get("status") != "SUCCESS":
                target_id = None
        else:
            for k, v in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', '')):
                if v.get("status") == "SUCCESS":
                    target_id = k
                    break
                    
        if not target_id:
            # 修改点：将 None 替换为 empty_image
            return (empty_image, "暂无下载任务，请运行查询节点刷新任务状态")

        task_info = tasks[target_id]
        url = task_info.get("image_url")
        
        if not url:
            # 修改点：将 None 替换为 empty_image
            return (empty_image, "任务成功但未找到 imageUrl")

        try:
            pil_image = download_image(url)
            if not pil_image:
                 raise Exception("下载网络超时或为空")
                 
            final_tensor = safe_pil_batch_to_tensor([pil_image])
            
            task_type = task_info.get("type", "")
            is_upscale = task_type.startswith("ACTION_U") 
            
            # 自动切分逻辑
            if "Yes" in split_grid and not is_upscale:
                b, h, w, c = final_tensor.shape
                h_mid = h // 2
                w_mid = w // 2
                top_left     = final_tensor[:, :h_mid, :w_mid, :]
                top_right    = final_tensor[:, :h_mid, w_mid:, :]
                bottom_left  = final_tensor[:, h_mid:, :w_mid, :]
                bottom_right = final_tensor[:, h_mid:, w_mid:, :]
                final_tensor = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
            
            mj_task_db.update_task(target_id, {"status": "DOWNLOADED"})
                    
            msg = f"下载成功并切分: {target_id}" if ("Yes" in split_grid and not is_upscale) else f"下载成功: {target_id}"
            return (final_tensor, msg)
            
        except Exception as e:
            # 修改点：将 None 替换为 empty_image
            return (empty_image, f"下载图片失败: {e}")

# --- 节点映射注册 ---
NODE_CLASS_MAPPINGS = {
    "MJImagineSubmit": MJImagineSubmit,
    "MJQueryTasks": MJQueryTasks,
    "MJActionSubmit": MJActionSubmit,
    "MJGetResult": MJGetResult,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MJImagineSubmit": "1. MJ 生图提交 (Imagine)",
    "MJQueryTasks": "2. MJ 全局查询 (Query)",
    "MJActionSubmit": "3. MJ 动作执行 (U/V/Reroll)",
    "MJGetResult": "4. MJ 获取出图 (Get Result)",
}
