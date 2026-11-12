"""
棋魂 - 界面长相学习与自测模块：Learn.py
====================================================================
职责（纯识别，不涉及任何点击）：
1. 扫描 samples/ 目录下各界面截图，用 OCR 提取每个界面独有的稳定特征词；
2. 将结果保存到 screen_features.json，支持手动删掉动态词，也支持配置 exclude_keywords 排除词；
3. 按“特征词命中率”打分（解决子弹窗界面词少、被底图模式选择词多抢占误判的问题）；
4. 支持 --verify 参数跑离线自测（不覆盖手动修改），支持 --live 参数实时监控。
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from Window import find_window, grab_screen

SAMPLES_DIR = "samples"
FEATURES_PATH = "screen_features.json"

_ocr_engine = RapidOCR()


def imread_cn(path: str) -> np.ndarray:
    """兼容中文路径读取图片"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_features(path: str = FEATURES_PATH) -> Dict[str, dict]:
    """读取已保存的界面指纹库"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def extract_tokens(img: np.ndarray) -> Set[str]:
    """
    从界面截图中提取干净的文字特征词集合。
    过滤掉纯数字（如变动中的金币、人数）和单字碎片。
    """
    if img is None or img.size == 0:
        return set()
    res, _ = _ocr_engine(img)
    if not res:
        return set()

    tokens: Set[str] = set()
    for _, text, conf in res:
        try:
            if float(conf) < 0.65:
                continue
        except (ValueError, TypeError):
            continue

        for part in re.split(r"[\s:：,，。!！?？【】\[\]()（）<>/|]+", text.strip()):
            part = part.strip()
            if len(part) >= 2 and not part.isdigit():
                tokens.add(part)
    return tokens


def learn_from_samples(samples_dir: str = SAMPLES_DIR, rebuild: bool = False) -> Dict[str, dict]:
    """
    从 samples/ 目录学习界面长相特征并保存到 screen_features.json。
    默认会保留你已在 screen_features.json 中手动精修过的界面，只补充新增截图。
    """
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"未找到 {samples_dir}/ 目录，请先用 Window.py 分开保存各界面截图。")

    files = sorted(f for f in os.listdir(samples_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise RuntimeError(f"{samples_dir}/ 目录为空。")

    raw_tokens: Dict[str, Set[str]] = {}
    doc_freq: Counter = Counter()

    print(f"=== 第1步：扫描 {samples_dir}/ 下的 {len(files)} 张界面样本 ===")
    for fname in files:
        screen_name = os.path.splitext(fname)[0]
        img = imread_cn(os.path.join(samples_dir, fname))
        tokens = extract_tokens(img)
        raw_tokens[screen_name] = tokens
        for t in tokens:
            doc_freq[t] += 1
        print(f"  [{screen_name}] 提取到 {len(tokens)} 个有效词：{sorted(tokens)}")

    existing = {} if rebuild else load_features()
    features: Dict[str, dict] = {}

    print("\n=== 第2步：提炼每个界面的独有长相指纹 ===")
    for screen_name, tokens in raw_tokens.items():
        if screen_name in existing:
            features[screen_name] = existing[screen_name]
            print(f"  [{screen_name}] 保留你已定义好的指纹：{features[screen_name]['unique_keywords']}")
            continue

        unique_words = sorted([t for t in tokens if doc_freq[t] == 1])
        if len(unique_words) < 2:
            unique_words = sorted(tokens, key=lambda t: (doc_freq[t], t))[:6]
        else:
            unique_words = unique_words[:8]

        features[screen_name] = {
            "unique_keywords": unique_words,
            "min_hits": min(2, len(unique_words)) if unique_words else 1,
        }
        print(f"  [{screen_name}] 自动提取指纹 ({len(unique_words)} 个)：{unique_words}")

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f"\n已保存至：{os.path.abspath(FEATURES_PATH)}")
    return features


def identify_screen(img: np.ndarray, features: Dict[str, dict]) -> Tuple[Optional[str], List[str]]:
    """
    根据 screen_features.json 判断一张图片属于哪个界面。
    采用“命中率 = 命中数 / 该界面总指纹数”打分，避免底图（如模式选择）词多抢占子界面（如房间码）。
    """
    tokens = extract_tokens(img)
    full_text = " ".join(tokens)

    best_name: Optional[str] = None
    best_matched: List[str] = []
    best_score: Tuple[float, int] = (0.0, 0)

    for screen_name, info in features.items():
        # 若画面中出现了该界面的“排除词”，直接跳过（用于严格区分父界面与弹窗子界面）
        excludes = info.get("exclude_keywords", [])
        if any(ex in full_text for ex in excludes):
            continue

        kws = info.get("unique_keywords", [])
        if not kws:
            continue

        need = info.get("min_hits", 2)
        matched = [kw for kw in kws if kw in full_text]
        if len(matched) >= need:
            # 优先比命中率，命中率相同时再比命中个数
            score = (len(matched) / len(kws), len(matched))
            if score > best_score:
                best_score = score
                best_name = screen_name
                best_matched = matched

    return best_name, best_matched


def verify_offline(features: Dict[str, dict], samples_dir: str = SAMPLES_DIR) -> None:
    """用 samples/ 下的原图做离线交叉验证，检验识别准确率"""
    files = sorted(f for f in os.listdir(samples_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    correct = 0

    print("\n=== 离线样本自测准确率 ===")
    for fname in files:
        expected = os.path.splitext(fname)[0]
        img = imread_cn(os.path.join(samples_dir, fname))
        pred, matched = identify_screen(img, features)
        ok = (pred == expected)
        correct += int(ok)
        mark = "正确" if ok else "误判"
        print(f"  [{mark}] 样本 {fname} -> 识别为：【{pred}】 (命中：{matched})")

    acc = (correct / len(files)) * 100 if files else 0.0
    print(f"\n>>> 自测总准确率：{correct}/{len(files)} ({acc:.1f}%)")


def watch_live() -> None:
    """使用 screen_features.json 实时监控游戏窗口"""
    features = load_features()
    if not features:
        raise FileNotFoundError(f"未找到 {FEATURES_PATH}，请先运行 python Learn.py 完成学习。")

    print("=== 实时界面识别模式已启动 (按 Ctrl+C 退出) ===")
    try:
        while True:
            rect = find_window()
            img = grab_screen(rect)
            pred, matched = identify_screen(img, features)
            print(f">>> 当前实时界面：【{pred}】 | 命中指纹：{matched}")
            time.sleep(1.2)
    except KeyboardInterrupt:
        print("\n已停止监控。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="棋魂 - 界面长相自动学习与自测")
    parser.add_argument("--verify", action="store_true", help="只用当前 screen_features.json 跑离线自测，不修改文件")
    parser.add_argument("--rebuild", action="store_true", help="忽略已有配置，全部重新自动提取")
    parser.add_argument("--live", action="store_true", help="使用学到的特征实时识别运行中的游戏画面")
    args = parser.parse_args()

    if args.live:
        watch_live()
    elif args.verify:
        verify_offline(load_features(), SAMPLES_DIR)
    else:
        learned = learn_from_samples(SAMPLES_DIR, rebuild=args.rebuild)
        verify_offline(learned, SAMPLES_DIR)