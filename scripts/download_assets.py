import os
import sys
import json
import requests
import shutil
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from configs.config import MODEL_CACHE_DIR
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config
from configs.config import MODEL_CACHE_DIR, MODEL_NAME

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_online(url="https://huggingface.co", timeout=5):
    try:
        requests.head(url, timeout=timeout)
        return True
    except Exception:
        return False

def download_repo(repo_id, target_dir):
    ensure_dir(target_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            force_download=False,
        )
    except Exception as e:
        raise RuntimeError(f"下载失败: {repo_id}: {e}")

def hub_repo_dir(repo_id):
    return os.path.join(MODEL_CACHE_DIR, f"models--{repo_id.replace('/', '--')}")

def purge_repo_cache(repo_id):
    d = hub_repo_dir(repo_id)
    if os.path.isdir(d):
        try:
            shutil.rmtree(d)
        except Exception:
            pass

def download_repo_to_cache(repo_id, force=False):
    try:
        required = [
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "source.spm",
            "target.spm",
            "spiece.model",
            "merges.txt",
            "README.md",
        ]
        for fname in required:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    cache_dir=MODEL_CACHE_DIR,
                    force_download=bool(force),
                    local_files_only=False,
                )
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"下载失败(缓存): {repo_id}: {e}")

def find_in_cache(repo_id, filename):
    base = hub_repo_dir(repo_id)
    if not os.path.isdir(base):
        return None
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower() == filename.lower():
                return os.path.join(root, f)
    return None

def ensure_flat_files_from_cache(repo_id, target_dir):
    ensure_dir(target_dir)
    for fname in ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json", "config.json", "generation_config.json", "pytorch_model.bin", "model.safetensors"]:
        dst = os.path.join(target_dir, fname)
        if os.path.exists(dst):
            continue
        src = find_in_cache(repo_id, fname)
        if src and os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

def predownload_t5():
    local_dir = os.path.join(MODEL_CACHE_DIR, MODEL_NAME)
    ensure_dir(local_dir)
    try:
        T5Config.from_pretrained(local_dir, local_files_only=True)
        T5ForConditionalGeneration.from_pretrained(local_dir, local_files_only=True)
        AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        return True
    except Exception:
        if not is_online():
            print(f"离线且本地缺少 {MODEL_NAME}，跳过下载。稍后联网再执行脚本即可。")
            return False
        download_repo(MODEL_NAME, local_dir)
        T5Config.from_pretrained(local_dir, local_files_only=True)
        T5ForConditionalGeneration.from_pretrained(local_dir, local_files_only=True)
        AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        return True

def predownload_marian(repo_id):
    local_dir = os.path.join(MODEL_CACHE_DIR, repo_id.replace("/", os.sep))
    ensure_dir(local_dir)
    required = [
        "config.json", "generation_config.json",
        "pytorch_model.bin", "model.safetensors",
        "tokenizer_config.json", "vocab.json", "source.spm", "target.spm"
    ]
    have_all = all(os.path.exists(os.path.join(local_dir, f)) for f in required)
    if have_all:
        try:
            AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
            AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)
            return True
        except Exception:
            try:
                AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True, use_safetensors=False)
                AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
                return True
            except Exception:
                pass
    if not is_online():
        print(f"离线且本地缺少 {repo_id} 的必要文件，跳过下载。稍后联网再执行脚本即可。")
        return False
    # 在线下载到平铺目录，避免快照路径缺失 spm
    download_repo(repo_id, local_dir)
    # 强制确保关键文件直接拉取到平铺目录（防止镜像不完整）
    for fname in ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json"]:
        if not os.path.exists(os.path.join(local_dir, fname)):
            try:
                path_in_cache = hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=MODEL_CACHE_DIR, force_download=True, local_files_only=False)
                if path_in_cache and os.path.exists(path_in_cache):
                    shutil.copy2(path_in_cache, os.path.join(local_dir, fname))
            except Exception:
                pass
    # 若平铺目录仍缺失关键文件，则从缓存快照中复制补齐
    ensure_flat_files_from_cache(repo_id, local_dir)
    # 再次以离线方式校验平铺目录
    try:
        AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)
    except Exception:
        AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True, use_safetensors=False)
    AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
    return True

def main():
    print(f"缓存目录: {MODEL_CACHE_DIR}")
    ensure_dir(MODEL_CACHE_DIR)
    ok_t5 = predownload_t5()
    ok_en_zh = predownload_marian("Helsinki-NLP/opus-mt-en-zh")
    ok_zh_en = predownload_marian("Helsinki-NLP/opus-mt-zh-en")
    print("下载状态：")
    print(f"- 摘要模型(t5-base): {'就绪' if ok_t5 else '未就绪'}")
    print(f"- 翻译(英->中): {'就绪' if ok_en_zh else '未就绪'}")
    print(f"- 翻译(中->英): {'就绪' if ok_zh_en else '未就绪'}")
    if ok_t5 and ok_en_zh and ok_zh_en:
        print("所有模型与分词器已下载到本地并可离线使用")
        return 0
    else:
        print("部分资源未就绪：联网后再次运行本脚本即可自动补齐")
        return 0

if __name__ == "__main__":
    sys.exit(main())

