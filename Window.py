"""
棋魂 - 第1块积木：Window.py
====================================================================
职责：
1. 通过模板匹配在桌面/任务栏找到「王者万象棋」图标并点击，将游戏唤醒到最前面；
2. 精准锁定游戏本体窗口（支持从最小化自动恢复，并自动排除浏览器/IDE标题干扰）；
3. 抓取画面后先 1:1 弹窗给你预览确认，确认无误后再让你命名保存（不满意可直接放弃保存）。
"""

import os
import time
from datetime import datetime

import cv2
import mss
import numpy as np
import pydirectinput
import pygetwindow as gw

GAME_TITLE = "王者万象棋"
TASKBAR_ICON = "templates/taskbar_icon.png"


def focus_by_icon(icon_path: str = TASKBAR_ICON, threshold: float = 0.8) -> bool:
    """在全屏（含任务栏）匹配游戏图标并点击唤醒窗口"""
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
    """精准查找游戏本体窗口的屏幕坐标与尺寸（若游戏被最小化会自动恢复）"""
    all_wins = gw.getWindowsWithTitle(title)
    exact = [w for w in all_wins if w.title.strip() == title]
    pool = exact if exact else [
        w for w in all_wins
        if not any(app in w.title for app in ("Chrome", "Edge", "Firefox", "PyCharm"))
    ]

    candidates = [w for w in pool if w.isMinimized or (w.width > 200 and w.height > 200)]
    if not candidates:
        raise RuntimeError(f"没找到「{title}」游戏窗口，请确认游戏已启动。")

    win = max(candidates, key=lambda w: w.width * w.height)
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
    """截取指定游戏窗口区域画面"""
    if rect is None:
        rect = find_window()
    with mss.MSS() as sct:
        raw = np.array(sct.grab(rect))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def save_sample(img: np.ndarray, name: str = "") -> str:
    """将截图分开保存到 samples/，支持中文文件名；若不指定名称则按时间戳命名"""
    os.makedirs("samples", exist_ok=True)
    if not name:
        name = datetime.now().strftime("shot_%Y%m%d_%H%M%S")
    path = os.path.join("samples", f"{name}.png")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("图像编码 PNG 失败。")
    buf.tofile(path)
    return path


if __name__ == "__main__":
    # 1. 唤醒游戏并抓拍当前画面
    focus_by_icon()
    rect = find_window()
    print(f"窗口坐标与尺寸：{rect}")
    img = grab_screen(rect)
    print(f"已抓取画面，尺寸：{img.shape[1]}x{img.shape[0]}，请先预览确认...")

    # 2. 先 1:1 弹窗展示抓到的画面，按任意键关闭预览后决定是否保存
    win_name = "Preview (Press any key to confirm & name)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3. 确认画面没问题后，再由你手动输入名称保存（输入 n 可放弃本次截图）
    ans = input("确认画面无误！请输入保存名称（如 对局中；输入 n 放弃保存；直接回车按时间命名）: ").strip()
    if ans.lower() in ("n", "no", "0"):
        print("已放弃保存本次截图。")
    else:
        saved_path = save_sample(img, ans)
        print(f"截图已保存至：{saved_path}")