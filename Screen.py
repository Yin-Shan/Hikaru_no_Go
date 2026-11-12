"""
棋魂 - 第4块积木：Screen.py
====================================================================
职责：
1. 读取 screens.json 界面地图，通过全窗关键词（长相特征）精准识别当前处于哪个界面；
2. 内置转场动画等待机制 (wait_for_screen)，过滤切屏动画期间的短暂 None 状态；
3. 根据界面跳转关系，用 BFS 最短路径算法规划点击顺序并自主导航。
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from Mouse import click_region, ensure_game_foreground
from Window import grab_screen

SCREENS_PATH = "screens.json"

# 全局初始化一次 OCR 引擎，避免重复加载模型耗时
_ocr_engine = RapidOCR()


def load_screens(path: str = SCREENS_PATH) -> dict:
    """读取界面知识地图 screens.json"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先将 screens.json 放到项目根目录。")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ocr_image(img: np.ndarray, scale: float = 1.0) -> str:
    """对图像做 OCR 并拼接返回所有识别到的文字"""
    if img.size == 0:
        return ""
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res, _ = _ocr_engine(img)
    if not res:
        return ""
    return " ".join(item[1] for item in res).strip()


def detect_screen(img: Optional[np.ndarray] = None, debug: bool = True) -> Optional[str]:
    """
    根据 screens.json 中每个界面的长相关键词，识别当前画面属于哪个界面。
    :param img: 游戏画面；不传则自动将游戏置顶后截取当前游戏窗口
    :param debug: 是否打印识别到的全窗文字和命中数，方便调试
    :return: 匹配到的界面名称；动画过渡期或都不匹配返回 None
    """
    screens = load_screens()

    if img is None:
        rect = ensure_game_foreground()
        img = grab_screen(rect)

    full_text = ocr_image(img, scale=1.0)
    if debug:
        print(f"[全窗文字] {full_text}")

    best_screen: Optional[str] = None
    best_hits = 0

    for screen_name, info in screens.items():
        keywords = info.get("keywords", [])
        if not keywords:
            continue

        hits = sum(1 for kw in keywords if kw in full_text)
        need = info.get("min_hits", min(2, len(keywords)))
        if debug:
            print(f"  [匹配统计] {screen_name}: 命中 {hits}/{len(keywords)} (至少需 {need})")

        if hits >= need and hits > best_hits:
            best_hits = hits
            best_screen = screen_name

    return best_screen


def wait_for_screen(timeout: float = 6.0, interval: float = 0.4) -> Optional[str]:
    """
    等待转场动画结束，持续轮询直到识别出有效界面（非 None）或超时。
    :param timeout: 最长等待秒数（默认 6 秒）
    :param interval: 每次检测间隔秒数
    :return: 稳定后的界面名称；超时仍为 None 则返回 None
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        screen = detect_screen(debug=False)
        if screen is not None:
            return screen
        print("  [动画过渡中] 暂未识别到稳定界面，继续等待...")
        time.sleep(interval)
    return None


def find_path(start: str, target: str) -> Optional[List[str]]:
    """使用 BFS 计算从当前界面到达目标界面的最短按钮点击序列"""
    if start == target:
        return []

    screens = load_screens()
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        cur, path = queue.popleft()
        actions: Dict[str, str] = screens.get(cur, {}).get("actions", {})
        for btn_name, next_screen in actions.items():
            if next_screen == target:
                return path + [btn_name]
            if next_screen not in visited and next_screen in screens:
                visited.add(next_screen)
                queue.append((next_screen, path + [btn_name]))
    return None


def navigate_to(target_screen: str, max_steps: int = 5) -> bool:
    """自主导航到目标界面：识别当前位置 -> 规划路径 -> 点击跳转 -> 等待动画结束并确认到达"""
    ensure_game_foreground()
    for _ in range(max_steps):
        current = wait_for_screen(timeout=6.0)
        print(f"[状态判断] 当前界面：{current} -> 目标界面：{target_screen}")

        if current == target_screen:
            print(f"已到达 [{target_screen}]！")
            return True
        if current is None:
            print("[警告] 等待超时仍无法识别当前界面，停止导航。")
            return False

        path = find_path(current, target_screen)
        if not path:
            print(f"[警告] 找不到从 [{current}] 到 [{target_screen}] 的路径。")
            return False

        next_btn = path[0]
        print(f"[自主规划] 路径={path} -> 点击 [{next_btn}]")
        click_region(next_btn)
        time.sleep(0.8)  # 点击后先缓冲 0.8 秒，随后由 wait_for_screen 自动等完动画

    return False


def watch_loop() -> None:
    """持续监控模式：自动置顶游戏后，你手动切换界面，控制台实时报出识别结果"""
    print("=== 界面长相监控模式已启动 (按 Ctrl+C 退出) ===")
    print("请在游戏里手动切换：大厅 -> 菜单 -> 大厅 -> 开始游戏(模式选择) -> 大厅\n")
    ensure_game_foreground()
    try:
        while True:
            screen = detect_screen(debug=True)
            print(f">>> 当前识别界面：【{screen}】\n")
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\n已退出监控。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="棋魂界面识别与自主导航")
    parser.add_argument("--watch", action="store_true", help="持续监控当前处于哪个界面")
    parser.add_argument("--goto", type=str, default="", help="测试自主导航到指定界面，如 --goto 模式选择")
    args = parser.parse_args()

    if args.watch:
        watch_loop()
    elif args.goto:
        navigate_to(args.goto)
    else:
        cur = detect_screen(debug=True)
        print(f"\n>>> 当前识别界面：【{cur}】")