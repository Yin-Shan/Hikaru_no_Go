"""
棋魂 - 第2块积木：Mark.py
====================================================================
职责：
1. 读取 samples/test_window.png（若无则调用 Window.py 现截一帧），让你用鼠标框出可点击区域并命名；
2. 将“区域名称 -> 相对窗口比例”保存到 regions.json（窗口拖动或缩放后依然精准）；
3. 在预览图上画出所有已保存的方框、中心准星和中文名称，方便肉眼核对位置是否准确。
"""

import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 与项目中的 Window.py 文件名保持一致（首字母大写）
from Window import find_window, grab_screen

SAMPLE_PATH = "samples/test_window.png"
REGIONS_PATH = "regions.json"


def load_regions(path: str = REGIONS_PATH) -> dict:
    """
    读取已保存的区域字典。
    :param path: JSON 配置文件路径
    :return: 字典 {区域名称: {"box_norm": [...], "click_norm": [...]}}
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_regions(regions: dict, path: str = REGIONS_PATH) -> None:
    """
    将区域字典以格式化中文 JSON 保存到磁盘。
    :param regions: 区域字典
    :param path: 保存路径
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regions, f, ensure_ascii=False, indent=2)


def load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """
    加载 Windows 系统中文字体，确保预览图上的中文名称能正常显示而不乱码。
    :param size: 字体大小（像素）
    :return: Pillow 字体对象
    """
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_regions(img: np.ndarray, regions: dict) -> np.ndarray:
    """
    把 regions.json 里的所有区域框、中心点击十字准星和中文名称画在图上。
    :param img: 原始游戏窗口图像 (BGR)
    :param regions: 区域字典
    :return: 叠加了标注框的图像 (BGR)
    """
    h, w = img.shape[:2]
    vis = img.copy()

    # 先用 OpenCV 绘制绿色矩形框和红色中心十字准星
    for info in regions.values():
        nx, ny, nw, nh = info["box_norm"]
        cnx, cny = info["click_norm"]
        x, y, bw, bh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
        cx, cy = int(cnx * w), int(cny * h)

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 220, 80), 2)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    # 再用 Pillow 渲染清晰的中文标签（OpenCV 原生 putText 不支持中文）
    pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(16)

    for name, info in regions.items():
        nx, ny, _, _ = info["box_norm"]
        x, y = int(nx * w), int(ny * h)
        draw.text((x, max(4, y - 22)), name, font=font, fill=(50, 255, 110))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def mark_interactive(img: np.ndarray) -> dict:
    """
    交互式循环框选可点击区域并输入名称。
    :param img: 用于框选的基准游戏画面
    :return: 更新后的完整区域字典
    """
    h, w = img.shape[:2]
    regions = load_regions()

    print("\n=== 开始标记可点击区域 ===")
    print("操作：鼠标左键框选区域 -> 按【Enter/空格】确认 -> 在控制台输入名称；按【ESC】结束标记。\n")

    win_name = "Mark Regions (Enter=Confirm, ESC=Finish)"
    while True:
        # 每次循环重绘已有标记，避免重复框选
        preview = draw_regions(img, regions)
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        box = cv2.selectROI(win_name, preview, showCrosshair=True, fromCenter=False)
        x, y, bw, bh = [int(v) for v in box]

        # 按 ESC 或框选宽高为 0 时结束标记
        if bw == 0 or bh == 0:
            break

        name = input("请为刚才框选的区域命名（如 开始游戏，直接回车取消该框）: ").strip()
        if not name:
            print("已取消该框。")
            continue

        cx, cy = x + bw // 2, y + bh // 2
        # 存为相对窗口宽高的比例 (0.0~1.0)，窗口拖动或调整大小后依然精准
        regions[name] = {
            "box_norm": [round(x / w, 5), round(y / h, 5), round(bw / w, 5), round(bh / h, 5)],
            "click_norm": [round(cx / w, 5), round(cy / h, 5)],
        }
        save_regions(regions)
        print(f"已保存 [{name}] -> 中心比例 {regions[name]['click_norm']}")

    cv2.destroyAllWindows()
    return regions


if __name__ == "__main__":
    # === 独立调试入口（在 PyCharm 右键 Run 'Mark' 即可单独验证） ===

    # 优先读取第1步保存的本地样本图；若不存在则自动调用 Window.py 现截一帧
    if os.path.exists(SAMPLE_PATH):
        base_img = cv2.imread(SAMPLE_PATH)
        print(f"已加载本地样本图：{SAMPLE_PATH}")
    else:
        rect = find_window()
        base_img = grab_screen(rect)

    all_regions = mark_interactive(base_img)
    print(f"\n当前共记录 {len(all_regions)} 个区域：{list(all_regions.keys())}")

    # 最终以 1:1 源分辨率弹出所有已标记区域供核对
    final_vis = draw_regions(base_img, all_regions)
    preview_win = "Saved Regions Preview (Press any key to close)"
    cv2.namedWindow(preview_win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(preview_win, final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()