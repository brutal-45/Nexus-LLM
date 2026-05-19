"""System info: CPU, GPU, RAM, disk, CUDA version, Python version, platform detection."""

import os
import sys
import platform
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information.

    Returns:
        Dictionary with system details including CPU, RAM, disk, Python, and OS info.
    """
    info = {
        "platform": get_platform_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "python": get_python_info(),
    }

    gpu_info = get_gpu_info()
    if gpu_info:
        info["gpu"] = gpu_info

    cuda_info = get_cuda_info()
    if cuda_info:
        info["cuda"] = cuda_info

    return info


def get_platform_info() -> Dict[str, str]:
    """Get operating system and platform information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "architecture": platform.architecture()[0],
    }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    info = {
        "count_logical": os.cpu_count() or 0,
        "count_physical": _get_physical_cpu_count(),
        "architecture": platform.machine(),
    }

    try:
        import multiprocessing
        info["count_logical"] = multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if line.startswith("model name"):
                    info["model"] = line.split(":")[1].strip()
                    break
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                info["model"] = result.stdout.strip()
        elif platform.system() == "Windows":
            info["model"] = platform.processor()
    except Exception:
        info["model"] = "Unknown"

    return info


def _get_physical_cpu_count() -> int:
    """Get the number of physical CPU cores."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 0
    except ImportError:
        pass

    try:
        if platform.system() == "Linux":
            with open("/sys/devices/system/cpu/present", "r") as f:
                core_range = f.read().strip()
                if "-" in core_range:
                    return int(core_range.split("-")[1]) + 1
    except Exception:
        pass

    return os.cpu_count() or 0


def get_memory_info() -> Dict[str, Any]:
    """Get RAM and memory information."""
    info = {}

    try:
        import psutil
        mem = psutil.virtual_memory()
        info = {
            "total_bytes": mem.total,
            "available_bytes": mem.available,
            "used_bytes": mem.used,
            "percent_used": mem.percent,
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
        }
    except ImportError:
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip().split()[0]
                            meminfo[key] = int(value) * 1024

                total = meminfo.get("MemTotal", 0)
                available = meminfo.get("MemAvailable", 0)
                info = {
                    "total_bytes": total,
                    "available_bytes": available,
                    "total_gb": round(total / (1024 ** 3), 2),
                    "available_gb": round(available / (1024 ** 3), 2),
                }
        except Exception:
            info = {"error": "Could not determine memory info"}

    return info


def get_disk_info(path: str = "/") -> Dict[str, Any]:
    """Get disk usage information.

    Args:
        path: Path to check disk usage for.

    Returns:
        Dictionary with disk usage details.
    """
    try:
        if platform.system() == "Windows":
            path = "C:\\"
        disk = os.statvfs(path) if hasattr(os, "statvfs") else None

        if disk:
            total = disk.f_blocks * disk.f_frsize
            free = disk.f_bavail * disk.f_frsize
            used = total - free
            return {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "percent_used": round(used / max(total, 1) * 100, 1),
            }
    except Exception:
        pass

    try:
        import shutil
        usage = shutil.disk_usage(path if platform.system() != "Windows" else "C:\\")
        return {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
            "percent_used": round(usage.used / max(usage.total, 1) * 100, 1),
        }
    except Exception:
        return {"error": "Could not determine disk info"}


def get_python_info() -> Dict[str, str]:
    """Get Python environment information."""
    import torch

    return {
        "version": sys.version,
        "version_short": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "executable": sys.executable,
        "prefix": sys.prefix,
        "torch_version": torch.__version__,
    }


def get_gpu_info() -> Optional[List[Dict[str, Any]]]:
    """Get GPU information for all available GPUs.

    Returns:
        List of GPU info dictionaries, or None if no GPUs.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                "index": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }

            try:
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                gpu_info["memory_allocated_bytes"] = allocated
                gpu_info["memory_reserved_bytes"] = reserved
                gpu_info["memory_allocated_gb"] = round(allocated / (1024 ** 3), 2)
            except Exception:
                pass

            gpus.append(gpu_info)

        return gpus
    except Exception:
        return None


def get_cuda_info() -> Optional[Dict[str, Any]]:
    """Get CUDA version information.

    Returns:
        Dictionary with CUDA details, or None if CUDA not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        return {
            "version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "num_gpus": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
        }
    except Exception:
        return None


def get_cuda_version() -> Optional[str]:
    """Get CUDA version as a string.

    Returns:
        CUDA version string, or None if not available.
    """
    info = get_cuda_info()
    if info:
        return info.get("version")
    return None


def print_system_info():
    """Print formatted system information to the logger."""
    info = get_system_info()

    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    plat = info.get("platform", {})
    logger.info(f"  OS: {plat.get('system', 'Unknown')} {plat.get('release', '')}")
    logger.info(f"  Machine: {plat.get('machine', 'Unknown')}")

    cpu = info.get("cpu", {})
    logger.info(f"  CPU: {cpu.get('model', 'Unknown')}")
    logger.info(f"  CPU Cores: {cpu.get('count_logical', 'Unknown')} logical, {cpu.get('count_physical', 'Unknown')} physical")

    mem = info.get("memory", {})
    logger.info(f"  RAM: {mem.get('total_gb', 'Unknown')} GB ({mem.get('percent_used', 'N/A')}% used)")

    gpu_list = info.get("gpu", [])
    if gpu_list:
        for gpu in gpu_list:
            logger.info(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
    else:
        logger.info("  GPU: None detected")

    cuda = info.get("cuda", {})
    if cuda:
        logger.info(f"  CUDA: {cuda.get('version', 'Unknown')}")
        if cuda.get("cudnn_version"):
            logger.info(f"  cuDNN: {cuda['cudnn_version']}")

    py = info.get("python", {})
    logger.info(f"  Python: {py.get('version_short', 'Unknown')}")
    logger.info(f"  PyTorch: {py.get('torch_version', 'Unknown')}")
    logger.info("=" * 60)
