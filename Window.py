"""
棋魂 - 第1块积木：Window.py
====================================================================
职责：
1. 通过模板匹配在桌面/任务栏找到「王者万象棋」图标并点击，将游戏唤醒到最前面；
2. 精准锁定游戏本体窗口（支持从最小化自动恢复，并自动排除浏览器/IDE标题干扰）；
3. 提供 1:1 源分辨率的固定预览窗口，确保后续坐标标记不失真。
"""

import os
import time
import cv2
import mss
import numpy as np
import pydirectinput
import pygetwindow as gw

# 游戏窗口标题关键字（需与窗口左上角标题一致）
GAME_TITLE = "王者万象棋"
# 任务栏图标模板路径（用于点击唤醒最小化或在后台的游戏）
TASKBAR_ICON = "templates/taskbar_icon.png"


def focus_by_icon(icon_path: str = TASKBAR_ICON, threshold: float = 0.8) -> bool:
    """
    在全屏（含任务栏）匹配游戏图标并点击唤醒窗口。
    :param icon_path: 图标模板图片路径
    :param threshold: 模板匹配相似度阈值（0~1）
    :return: 成功匹配并点击返回 True，否则返回 False
    """
    if not os.path.exists(icon_path):
        print(f"[提示] 未找到图标模板 {icon_path}，跳过图标点击。")
        return False

    template = cv2.imread(icon_path, cv2.IMREAD_COLOR)
    h, w = template.shape[:2]

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        screen = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)

    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        print(f"[提示] 未在屏幕上匹配到游戏图标 (最高相似度 {max_val:.2f})")
        return False

    cx = monitor["left"] + max_loc[0] + w // 2
    cy = monitor["top"] + max_loc[1] + h // 2
    print(f"识别到任务栏图标 (相似度 {max_val:.2f})，点击 ({cx}, {cy}) 唤醒游戏...")
    pydirectinput.moveTo(cx, cy)
    pydirectinput.click()
    time.sleep(0.6)
    return True


def find_window(title: str = GAME_TITLE) -> dict:
    """
    精准查找游戏本体窗口的屏幕坐标与尺寸（若游戏被最小化会自动恢复）。
    :param title: 目标窗口标题
    :return: 字典 {"left": 左上角X, "top": 左上角Y, "width": 宽, "height": 高}
    """
    all_wins = gw.getWindowsWithTitle(title)

    # 优先精确匹配窗口标题，避免误抓到标题里也包含“王者万象棋”的浏览器或 IDE
    exact = [w for w in all_wins if w.title.strip() == title]
    pool = exact if exact else [
        w for w in all_wins
        if not any(app in w.title for app in ("Chrome", "Edge", "Firefox", "PyCharm"))
    ]

    # 保留“正常大小窗口”或“处于最小化状态的窗口”（最小化时系统报告的宽高小于 200，不能提前过滤掉）
    candidates = [w for w in pool if w.isMinimized or (w.width > 200 and w.height > 200)]
    if not candidates:
        raise RuntimeError(f"没找到「{title}」游戏窗口，请确认游戏已启动。")

    win = max(candidates, key=lambda w: w.width * w.height)

    # 如果游戏窗口处于最小化状态，先恢复显示，再读取恢复后的真实坐标和尺寸
    if win.isMinimized:
        win.restore()
        time.sleep(0.5)

    print(f"已锁定窗口标题：[{win.title}]")
    return {
        "left": int(win.left),
        "top": int(win.top),
        "width": int(win.width),
        "height": int(win.height),
    }


def grab_screen(rect: dict = None) -> np.ndarray:
    """
    截取指定游戏窗口区域画面。
    :param rect: 窗口坐标字典；若不传则自动调用 find_window() 获取
    :return: OpenCV 格式的 BGR 图像数组
    """
    if rect is None:
        rect = find_window()
    with mss.MSS() as sct:
        raw = np.array(sct.grab(rect))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


if __name__ == "__main__":
    focus_by_icon()

    rect = find_window()
    print(f"窗口坐标与尺寸：{rect}")
    img = grab_screen(rect)

    os.makedirs("samples", exist_ok=True)
    cv2.imwrite("samples/test_window.png", img)
    print(f"截图已保存至 samples/test_window.png，尺寸：{img.shape[1]}x{img.shape[0]}")

    win_name = "QiHun View (Press any key to close)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()