"""Refuse to start GPU mode when llama.cpp cannot see a Vulkan GPU."""
from pathlib import Path
import os
import re
import subprocess
import sys


def detect_devices():
    import mineru_llama_cpp

    binary = Path(mineru_llama_cpp.__file__).parent / 'bin' / 'llama-server'
    result = subprocess.run([str(binary), '--list-devices'], capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('llama.cpp 设备检查失败：' + result.stderr.strip()[-1500:])
    devices = re.findall(r'^\s*Vulkan\d+:\s*(.+)$', result.stdout, re.MULTILINE)
    if not devices:
        raise RuntimeError('llama.cpp 未检测到 Vulkan GPU。请确认 NVIDIA 驱动、Container Toolkit、'
                           'nvidia 容器运行时以及 graphics 驱动能力可用，并使用 compose.gpu.yaml 启动。')
    return devices


def main():
    try:
        devices = detect_devices()
    except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print('GPU 启动检查失败：' + str(error), file=sys.stderr, flush=True)
        return 1
    print('GPU 推理已启用（llama.cpp / Vulkan）：' + '；'.join(devices), flush=True)
    print('版面、OCR 等 ONNX 小模型继续使用 CPU。', flush=True)
    if len(sys.argv) > 1:
        os.execvp(sys.argv[1], sys.argv[1:])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
