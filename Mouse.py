"""
棋魂 - 第3块积木：Mouse.py
====================================================================
职责：
1. 点击前自动检查并确保「王者万象棋」已置顶到最前面（防止误点到最大化中的 PyCharm）；
2. 读取 regions.json 中记录的“区域名称 -> 框选范围”；
3. 每次点击时在框选区域内部随机挑选不同落点，并用贝塞尔曲线平滑点击。
"""

import json
import os
import random
import time
from typing import Tuple

import numpy as np
import pydirectinput
import pygetwindow as gw

# 与项目中的 Window.py 保持一致（首字母大写）
from Window import GAME_TITLE, find_window, focus_by_icon

# 关闭 pydirectinput 默认每步 0.1 秒停顿，使贝塞尔曲线轨迹流畅
pydirectinput.PAUSE = 0.0

REGIONS_PATH = "regions.json"


def load_regions(path: str = REGIONS_PATH) -> dict:
    """读取 regions.json 区域配置"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先运行 Mark.py 标记区域。")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_game_foreground(title: str = GAME_TITLE) -> dict:
    """
    确保游戏窗口当前处于最前面（激活状态），再返回其坐标，避免点到背后的 PyCharm。
    :param title: 游戏窗口标题
    :return: 游戏窗口坐标字典 {"left", "top", "width", "height"}
    """
    active = gw.getActiveWindow()
    if active and active.title.strip() == title:
        return find_window(title)

    # 方法1：先尝试点击任务栏图标唤醒游戏
    focus_by_icon()
    time.sleep(0.3)
    active = gw.getActiveWindow()
    if active and active.title.strip() == title:
        return find_window(title)

    # 方法2：若图标未匹配到，则通过系统窗口句柄强制将游戏置顶
    wins = [w for w in gw.getWindowsWithTitle(title) if w.title.strip() == title and w.width > 200]
    if not wins:
        raise RuntimeError(f"未找到「{title}」游戏窗口，无法执行点击。")

    win = wins[0]
    if win.isMinimized:
        win.restore()
        time.sleep(0.3)

    # Windows 下轻按一下 Alt 可解除后台程序无法抢占前台焦点的限制
    pydirectinput.press("alt")
    try:
        win.activate()
    except Exception:
        pass
    time.sleep(0.4)

    # 再次核对是否真的切到了游戏窗口，防止误点其他软件
    active = gw.getActiveWindow()
    if not active or active.title.strip() != title:
        raise RuntimeError("未能将「王者万象棋」切到最前面，已中止点击以防误触其他窗口。")

    return find_window(title)


def pick_random_point(box_norm: list, rect: dict, margin_ratio: float = 0.15) -> Tuple[int, int]:
    """
    在框选矩形内部随机选取一个屏幕绝对坐标点，保证每次落点不同。
    :param box_norm: 归一化框选区域 [nx, ny, nw, nh]
    :param rect: 当前游戏窗口位置与尺寸
    :param margin_ratio: 内缩安全边距比例（默认 15%），防止贴边误触
    :return: 屏幕绝对坐标 (abs_x, abs_y)
    """
    nx, ny, nw, nh = box_norm
    win_x, win_y = rect["left"], rect["top"]
    win_w, win_h = rect["width"], rect["height"]

    x1 = win_x + nx * win_w
    y1 = win_y + ny * win_h
    bw = nw * win_w
    bh = nh * win_h

    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio
    safe_left = int(x1 + pad_x)
    safe_right = max(safe_left + 1, int(x1 + bw - pad_x))
    safe_top = int(y1 + pad_y)
    safe_bottom = max(safe_top + 1, int(y1 + bh - pad_y))

    return random.randint(safe_left, safe_right), random.randint(safe_top, safe_bottom)


def move_human(target_x: int, target_y: int, steps: int = 16) -> None:
    """
    使用三次贝塞尔曲线将鼠标平滑移动到目标坐标，模拟人手弧线。
    """
    p0 = pydirectinput.position()
    p3 = (target_x, target_y)
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]

    p1 = (p0[0] + dx // 3 + random.randint(-35, 35), p0[1] + dy // 3 + random.randint(-35, 35))
    p2 = (p0[0] + 2 * dx // 3 + random.randint(-35, 35), p0[1] + 2 * dy // 3 + random.randint(-35, 35))

    for t in np.linspace(0.0, 1.0, steps):
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        pydirectinput.moveTo(int(x), int(y))
        time.sleep(random.uniform(0.004, 0.012))


def click_region(name: str) -> Tuple[int, int]:
    """
    核心指令：自动把游戏切到最前面，并在指定区域的框选范围内随机挑一点点击。
    :param name: regions.json 中记录的区域名称（如 "开始游戏"、"交易行"、"菜单"）
    :return: 本次实际点击的屏幕坐标 (tx, ty)
    """
    regions = load_regions()
    if name not in regions:
        raise KeyError(f"regions.json 中没有找到区域 [{name}]，已有区域：{list(regions.keys())}")

    # 每次点击前强制确认游戏窗口在最前，防止点到 PyCharm
    rect = ensure_game_foreground()

    tx, ty = pick_random_point(regions[name]["box_norm"], rect)
    print(f"执行指令：点击 [{name}] -> 本次随机落点 ({tx}, {ty})")

    move_human(tx, ty)
    time.sleep(random.uniform(0.04, 0.09))
    pydirectinput.mouseDown()
    time.sleep(random.uniform(0.04, 0.08))
    pydirectinput.mouseUp()
    return tx, ty


if __name__ == "__main__":
    # === 独立调试入口：即使 PyCharm 最大化，输入名称后也会自动切出游戏再点击 ===
    target_name = input("请输入要测试点击的区域名称（如 菜单 / 交易行 / 开始游戏）: ").strip()
    if target_name:
        click_region(target_name)