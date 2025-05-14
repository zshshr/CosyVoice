# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import logging
import platform
import psutil
import GPUtil
import onnxruntime as ort
import threading
import time


# 仅做日志提示，不调用 set_default_providers
if 'CUDAExecutionProvider' in ort.get_available_providers():
    logging.info("Using CUDA for onnxruntime")
    # 添加 onnxruntime 优化选项
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    session_options.log_severity_level = 1
else:
    logging.warning("CUDA not available for onnxruntime, using CPU")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    session_options.log_severity_level = 1

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# 检查 ttsfrd 是否可用
try:
    import ttsfrd
    TTSFRD_AVAILABLE = True
except ImportError:
    TTSFRD_AVAILABLE = False
    logging.warning("ttsfrd package not available, using WeTextProcessing instead")

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 模型配置
AVAILABLE_MODELS = {
    'CosyVoice-300M': 'pretrained_models/CosyVoice-300M',
    'CosyVoice-300M-SFT': 'pretrained_models/CosyVoice-300M-SFT',
    'CosyVoice-300M-Instruct': 'pretrained_models/CosyVoice-300M-Instruct',
    'CosyVoice2-0.5B': 'pretrained_models/CosyVoice2-0.5B'
}

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

def check_ttsfrd_installation():
    """检查 ttsfrd 安装状态并提供建议"""
    if not TTSFRD_AVAILABLE:
        system = platform.system()
        if system == "Windows":
            return """
            ttsfrd 包未安装或不可用。当前系统为 Windows，但提供的 ttsfrd wheel 文件仅支持 Linux。
            系统将使用 WeTextProcessing 作为替代。
            
            注意：这不会影响基本功能，但可能会影响某些文本处理特性。
            """
        else:
            return """
            ttsfrd 包未安装。请运行以下命令安装：
            pip install pretrained_models/CosyVoice-ttsfrd/ttsfrd_dependency-0.1-py3-none-any.whl
            pip install pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
            """
    return "ttsfrd 包已正确安装"

def load_model(model_name):
    """加载指定的模型"""
    model_path = AVAILABLE_MODELS.get(model_name)
    if not model_path:
        return None, "模型不存在"
    
    if not os.path.exists(model_path):
        return None, f"模型目录 {model_path} 不存在，请先下载模型"
    
    try:
        logging.info(f"Attempting to load model from {model_path}")
        # 根据模型类型选择加载方式
        if model_name == 'CosyVoice2-0.5B':
            model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=True)
        else:
            model = CosyVoice(model_path, load_jit=False, load_trt=False, fp16=False)
        
        logging.info("Successfully loaded model")
        device_info = "CPU" if not torch.cuda.is_available() else f"GPU ({torch.cuda.get_device_name(0)})"
        
        # 获取预训练音色列表
        spk_list = model.list_available_spks()
        logging.info(f"Available speakers: {spk_list}")
        
        return model, f"模型加载成功 (使用{device_info}进行推理)"
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        error_msg = f"模型加载失败: {str(e)}"
        if "CUDA" in str(e):
            error_msg += "\n注意：未检测到CUDA支持，将使用CPU进行推理，这可能会影响性能"
        return None, error_msg

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    global cosyvoice  # 添加全局变量声明
    
    if current_model[0] is None:
        gr.Warning('请先加载模型')
        return (16000, np.zeros(16000))
    
    cosyvoice = current_model[0]  # 使用当前加载的模型
    
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # 检查输入参数
    if not tts_text:
        gr.Warning('请输入要合成的文本')
        return (16000, np.zeros(16000))

    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.instruct is False:
            gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            return (16000, np.zeros(16000))
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            return (16000, np.zeros(16000))
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')

    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
            return (16000, np.zeros(16000))
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            return (16000, np.zeros(16000))
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')

    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            return (16000, np.zeros(16000))
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (16000, np.zeros(16000))

    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            return (16000, np.zeros(16000))

    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            return (16000, np.zeros(16000))
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    try:
        if mode_checkbox_group == '预训练音色':
            logging.info('get sft inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '3s极速复刻':
            logging.info('get zero_shot inference request')
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '跨语种复刻':
            logging.info('get cross_lingual inference request')
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        else:
            logging.info('get instruct inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    except Exception as e:
        logging.error(f"Error during audio generation: {str(e)}")
        gr.Error(f"生成失败: {str(e)}")
        return (16000, np.zeros(16000))

def get_system_info():
    """获取系统硬件信息"""
    info = []
    
    # CPU信息
    cpu_info = f"CPU: {platform.processor()}"
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_info += f" ({cpu_count} 物理核心, {cpu_logical} 逻辑核心)"
    info.append(cpu_info)
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_info = f"内存: {memory.total / (1024**3):.1f}GB (可用: {memory.available / (1024**3):.1f}GB)"
    info.append(memory_info)
    
    # CUDA信息
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_info = f"CUDA: 可用 (版本 {torch.version.cuda})"
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu = GPUtil.getGPUs()[i]
            gpu_info.append(f"GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
        info.append(cuda_info)
        info.extend(gpu_info)
    else:
        info.append("CUDA: 不可用 (将使用CPU进行推理)")
        info.append("注意：使用CPU进行推理可能会影响性能")
    
    # 操作系统信息
    system = platform.system()
    
    # 检测 Windows 11
    if system == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                product_name = winreg.QueryValueEx(key, "ProductName")[0]
                build_number = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
                if "Windows 11" in product_name or int(build_number) >= 22000:
                    os_info = f"操作系统: Windows 11 (Build {build_number})"
                else:
                    os_info = f"操作系统: {product_name} (Build {build_number})"
        except:
            # 如果无法获取详细信息，使用基本系统信息
            os_info = f"操作系统: {system} {platform.release()}"
    else:
        os_info = f"操作系统: {system} {platform.release()}"
    
    info.append(os_info)
    
    return "\n".join(info)

def main():
    # 全局变量声明
    global current_model, cosyvoice, prompt_sr
    current_model = [None]  # 初始化为空列表
    prompt_sr = 16000  # 设置采样率

    # 读取使用说明文档
    usage_guide = ""
    try:
        with open('USAGE.md', 'r', encoding='utf-8') as f:
            usage_guide = f.read()
    except Exception as e:
        logging.error(f"Failed to read USAGE.md: {str(e)}")
        usage_guide = "使用说明文档加载失败，请检查 USAGE.md 文件是否存在。"

    with gr.Blocks() as demo:
        gr.Markdown("### CosyVoice 语音合成系统")
        
        # 系统信息显示
        with gr.Accordion("系统信息", open=True):
            system_info = get_system_info()
            gr.Markdown(f"```\n{system_info}\n```")
        
        # ttsfrd 状态检查
        ttsfrd_status = check_ttsfrd_installation()
        if not TTSFRD_AVAILABLE:
            with gr.Accordion("⚠️ 系统提示", open=True):
                gr.Markdown(ttsfrd_status)
        
        # 模型选择和加载部分
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                label="选择模型",
                value=list(AVAILABLE_MODELS.keys())[0]
            )
            load_model_button = gr.Button("加载模型")
            model_status = gr.Textbox(label="模型状态", interactive=False)
    
        
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=2)
            sft_dropdown = gr.Dropdown(
                choices=[''],  # 初始化为空列表，但包含一个空选项
                label='选择预训练音色',
                value='',  # 初始值设为空字符串
                scale=1
            )
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=1):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        def on_model_load(model_name):
            """处理模型加载事件"""
            try:
                model, status = load_model(model_name)
                if model is not None:
                    current_model[0] = model
                    # 获取预训练音色列表
                    spk_list = model.list_available_spks()
                    logging.info(f"Available speakers for {model_name}: {spk_list}")
                    
                    if not spk_list:
                        logging.warning(f"No speakers found for model {model_name}")
                        return status, gr.update(choices=[''], value='')
                    
                    return status, gr.update(choices=spk_list, value=spk_list[0])
                else:
                    current_model[0] = None
                    return status, gr.update(choices=[''], value='')
            except Exception as e:
                logging.error(f"Error in on_model_load: {str(e)}")
                return f"加载模型时发生错误: {str(e)}", gr.update(choices=[''], value='')

        def on_generate_audio(*args):
            if current_model[0] is None:
                gr.Warning('请先加载模型')
                return (16000, np.zeros(16000))
            
            try:
                for audio in generate_audio(*args):
                    if isinstance(audio, tuple) and len(audio) == 2:
                        sample_rate, audio_data = audio
                        if isinstance(audio_data, np.ndarray):
                            return (sample_rate, audio_data)
                    return (16000, np.zeros(16000))
            except Exception as e:
                logging.error(f"Error during audio generation: {str(e)}")
                gr.Error(f"生成失败: {str(e)}")
                return (16000, np.zeros(16000))

        # 在页面底部添加详细使用说明
        with gr.Accordion("📖 详细使用说明", open=False):
            gr.Markdown(usage_guide)

        # 绑定事件
        load_model_button.click(
            on_model_load,
            inputs=[model_dropdown],
            outputs=[model_status, sft_dropdown]
        )
        
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            on_generate_audio,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                    seed, stream, speed],
            outputs=[audio_output]
        )
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])

    # 注释掉 queue 相关代码
    # demo.queue(max_size=4, default_concurrency_limit=2)
    
    # 直接启动 Gradio，不使用线程，但允许后台运行
    demo.launch(
        server_name='127.0.0.1', 
        server_port=args.port, 
        inbrowser=False, 
        show_error=True,
        prevent_thread_lock=True  # 添加这个参数让 Gradio 在后台运行
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    # 在主线程中启动 Gradio
    main()

    # 等待 Gradio 服务器启动
    import requests
    import time
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{args.port}")
            if r.status_code == 200:
                break
        except:
            time.sleep(1)

    # 在主线程中启动 webview
    import webview
    webview.create_window(
        "CosyVoice 桌面版", 
        f"http://127.0.0.1:{args.port}", 
        width=1280, 
        height=900, 
        resizable=True, 
        background_color="#f3f3f3"
    )
    webview.start()
