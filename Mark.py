"""
棋魂 - 第2块积木：Mark.py
====================================================================
职责：
1. 自动扫描 samples/ 下分开保存的各界面截图，让你选择要标记哪一张（或现截一帧）；
2. 自动记录当前标记属于哪个界面 (screen)，只显示当前界面自己的框，避免大厅和菜单的框混在一起；
3. 将“区域名称 -> 相对窗口比例”保存到 regions.json，方便 Mouse.py 读取点击。
"""

import json
import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Window import find_window, grab_screen

SAMPLES_DIR = "samples"
REGIONS_PATH = "regions.json"


def load_regions(path: str = REGIONS_PATH) -> dict:
    """读取已保存的区域字典"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_regions(regions: dict, path: str = REGIONS_PATH) -> None:
    """将区域字典以格式化中文 JSON 保存到磁盘"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regions, f, ensure_ascii=False, indent=2)


def load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """加载 Windows 系统中文字体，确保预览图上的中文名称不乱码"""
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def imread_cn(path: str) -> np.ndarray:
    """兼容中文路径的图片读取"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def choose_sample_image() -> Tuple[np.ndarray, str]:
    """
    列出 samples/ 目录下的所有界面截图供选择。
    :return: (图像数组, 当前界面名称)
    """
    files = []
    if os.path.isdir(SAMPLES_DIR):
        files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg")))

    if not files:
        print("samples/ 目录下暂无截图，正在直接截取当前游戏窗口...")
        return grab_screen(find_window()), "临时截图"

    print("\n=== 请选择要标记的界面截图 ===")
    print("  [0] 现截一张当前游戏窗口")
    for idx, fname in enumerate(files, start=1):
        print(f"  [{idx}] {fname}")

    choice = input("请输入序号（直接回车默认选最后一张）: ").strip()
    if choice == "0":
        return grab_screen(find_window()), "临时截图"
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        fname = files[int(choice) - 1]
    else:
        fname = files[-1]

    target = os.path.join(SAMPLES_DIR, fname)
    screen_name = os.path.splitext(fname)[0]
    print(f"已加载截图：{target}（所属界面：【{screen_name}】）")
    return imread_cn(target), screen_name


def draw_regions(img: np.ndarray, regions: dict, current_screen: str) -> np.ndarray:
    """
    只把属于 current_screen 的区域框、中心十字和中文名称画在图上，避免跨界面串框。
    （兼容早期未写 screen 字段的旧数据，默认归为“大厅”）
    """
    h, w = img.shape[:2]
    vis = img.copy()

    filtered = {
        name: info
        for name, info in regions.items()
        if info.get("screen", "大厅") == current_screen
    }

    for info in filtered.values():
        nx, ny, nw, nh = info["box_norm"]
        cnx, cny = info["click_norm"]
        x, y, bw, bh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
        cx, cy = int(cnx * w), int(cny * h)

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 220, 80), 2)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(16)

    for name, info in filtered.items():
        nx, ny, _, _ = info["box_norm"]
        x, y = int(nx * w), int(ny * h)
        draw.text((x, max(4, y - 22)), name, font=font, fill=(50, 255, 110))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def mark_interactive(img: np.ndarray, current_screen: str) -> dict:
    """交互式循环框选可点击区域并输入名称"""
    h, w = img.shape[:2]
    regions = load_regions()

    print(f"\n=== 开始标记【{current_screen}】的可点击区域 ===")
    print("操作：鼠标左键框选区域 -> 按【Enter/空格】确认 -> 在控制台输入名称；按【ESC】结束标记。\n")

    win_name = f"Mark [{current_screen}] (Enter=Confirm, ESC=Finish)"
    while True:
        preview = draw_regions(img, regions, current_screen)
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        box = cv2.selectROI(win_name, preview, showCrosshair=True, fromCenter=False)
        x, y, bw, bh = [int(v) for v in box]

        if bw == 0 or bh == 0:
            break

        name = input("请为刚才框选的区域命名（如 菜单_返回，直接回车取消该框）: ").strip()
        if not name:
            print("已取消该框。")
            continue

        cx, cy = x + bw // 2, y + bh // 2
        regions[name] = {
            "screen": current_screen,
            "box_norm": [round(x / w, 5), round(y / h, 5), round(bw / w, 5), round(bh / h, 5)],
            "click_norm": [round(cx / w, 5), round(cy / h, 5)],
        }
        save_regions(regions)
        print(f"已保存 [{name}] (所属界面: {current_screen}) -> 中心比例 {regions[name]['click_norm']}")

    cv2.destroyAllWindows()
    return regions


if __name__ == "__main__":
    base_img, cur_screen = choose_sample_image()
    all_regions = mark_interactive(base_img, cur_screen)

    cur_keys = [k for k, v in all_regions.items() if v.get("screen", "大厅") == cur_screen]
    print(f"\n【{cur_screen}】当前已记录 {len(cur_keys)} 个区域：{cur_keys}")

    final_vis = draw_regions(base_img, all_regions, cur_screen)
    preview_win = f"Saved [{cur_screen}] Preview (Press any key to close)"
    cv2.namedWindow(preview_win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(preview_win, final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()