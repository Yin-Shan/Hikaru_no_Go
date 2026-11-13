"""
棋魂 - 自主探索与疑问汇报模块：Explore.py
====================================================================
职责（第3步：自己探索界面关联 + 把不懂的地方存进 unknown/ 问你）：
1. 识别当前所处界面 (Learn.py)；
2. 自动扫描当前画面上的可点击文字按钮（内置安全黑名单，绝不乱点首充/祈愿/商城消费）；
3. 试探点击按钮并观察跳转结果，自动写入 screen_graph.json；
4. 遇到“陌生界面”或“无效点击”时，自动截图存进 unknown/ 并写入 questions.json 向你提问；
5. 支持 --tour 按学会的地图自主巡航，支持 --auto 让棋魂自动探索。
"""

import argparse
import json
import os
import random
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydirectinput
from rapidocr_onnxruntime import RapidOCR

from Learn import FEATURES_PATH, identify_screen, load_features
from Mouse import click_region, ensure_game_foreground, load_regions, move_human
from Window import grab_screen

GRAPH_PATH = "screen_graph.json"
UNKNOWN_DIR = "unknown"
QUESTIONS_PATH = os.path.join(UNKNOWN_DIR, "questions.json")

# 安全黑名单：自主探索时绝对禁止点击的敏感按钮关键词
DANGER_KEYWORDS = ("首充", "充值", "祈愿", "商城", "交易行", "购买", "支付", "退出", "注销", "删除")

_ocr_engine = RapidOCR()


def load_graph(path: str = GRAPH_PATH) -> Dict[str, Dict[str, str]]:
    """读取已学会的界面跳转关系图 {当前界面: {按钮名: 目标界面}}"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_graph(graph: Dict[str, Dict[str, str]], path: str = GRAPH_PATH) -> None:
    """保存界面跳转关系图"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def imwrite_cn(path: str, img: np.ndarray) -> None:
    """兼容中文路径保存图片"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"保存图片失败：{path}")
    buf.tofile(path)


def record_question(q_type: str, question: str, img: Optional[np.ndarray] = None, extra: dict = None) -> str:
    """
    将棋魂“看不懂的问题”与现场截图保存到 unknown/ 疑问收集箱，方便你后续统一查看解答。
    """
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = ""
    if img is not None:
        img_path = os.path.join(UNKNOWN_DIR, f"{q_type}_{ts}.png")
        imwrite_cn(img_path, img)

    questions = []
    if os.path.exists(QUESTIONS_PATH):
        with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)

    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": q_type,
        "question": question,
        "image": img_path,
    }
    if extra:
        entry.update(extra)

    questions.append(entry)
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"  [已记入错题本] {q_type}: {question}")
    if img_path:
        print(f"  [现场截图] {img_path}")
    return img_path


def scan_clickable_texts(img: np.ndarray, rect: dict) -> List[Dict[str, object]]:
    """
    用 OCR 自动扫描当前画面上的文字按钮位置（已过滤掉安全黑名单中的危险按钮）。
    返回 [{"text": 文字, "abs_xy": (屏幕X, 屏幕Y)}, ...]
    """
    res, _ = _ocr_engine(img)
    if not res:
        return []

    found = []
    for box, text, conf in res:
        try:
            if float(conf) < 0.75:
                continue
        except (ValueError, TypeError):
            continue

        clean = text.strip()
        # 过滤太短的单字、纯数字以及危险消费词汇
        if len(clean) < 2 or clean.isdigit():
            continue
        if any(bad in clean for bad in DANGER_KEYWORDS):
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        # 在文字框内加微小随机偏移，避免点死同一个像素
        rx = int(random.uniform(min(xs) + 2, max(xs) - 2)) if max(xs) - min(xs) > 6 else int(sum(xs) / 4)
        ry = int(random.uniform(min(ys) + 2, max(ys) - 2)) if max(ys) - min(ys) > 6 else int(sum(ys) / 4)
        found.append({
            "text": clean,
            "abs_xy": (rect["left"] + rx, rect["top"] + ry),
        })
    return found


def click_abs_human(abs_xy: Tuple[int, int], rect: dict) -> None:
    """用拟人变速曲线平滑移动并点击屏幕绝对坐标"""
    move_human(abs_xy[0], abs_xy[1], rect=rect)
    time.sleep(random.uniform(0.04, 0.09))
    pydirectinput.mouseDown()
    time.sleep(random.uniform(0.04, 0.08))
    pydirectinput.mouseUp()


def wait_stable_screen(features: dict, timeout: float = 5.0, interval: float = 0.4) -> Tuple[Optional[str], np.ndarray]:
    """等待转场动画结束，返回 (识别到的界面名或None, 最后一帧图像)"""
    deadline = time.time() + timeout
    last_img = None
    while time.time() < deadline:
        rect = ensure_game_foreground()
        last_img = grab_screen(rect)
        screen_name, _ = identify_screen(last_img, features)
        if screen_name is not None:
            return screen_name, last_img
        print("  [动画过渡中] 等待画面稳定...")
        time.sleep(interval)
    return None, last_img


def explore_auto(max_steps: int = 8) -> None:
    """
    全自动有边界探索模式：
    棋魂自己看当前界面 -> 挑一个没试过的安全按钮点一下 -> 认新界面并记图谱；
    遇到不认识的界面或无效点击，自动拍照存进 unknown/ 问你。
    """
    features = load_features(FEATURES_PATH)
    if not features:
        raise FileNotFoundError("请先运行 Learn.py 生成 screen_features.json。")

    regions = load_regions()
    graph = load_graph()
    tried_deadends: set = set()  # 记录点完无反应的 (界面, 按钮)

    print(f"=== 棋魂：自主探索模式启动（最多探索 {max_steps} 步，安全黑名单已开启）===")
    for step in range(1, max_steps + 1):
        print(f"\n--- 探索第 {step}/{max_steps} 步 ---")
        cur_screen, cur_img = wait_stable_screen(features)

        if cur_screen is None:
            record_question(
                q_type="陌生界面",
                question="当前画面无法匹配到任何已知界面，请帮我命名并移入 samples/ 学习。",
                img=cur_img,
            )
            print("探索暂停：遇到陌生界面已记录到 unknown/，请手动处理后再继续。")
            break

        # 收集本界面已知按钮（regions.json 中属于当前界面的）+ 自动扫描到的文字按钮
        known_btns = [
            name for name, info in regions.items()
            if info.get("screen", "大厅") == cur_screen
            and not any(bad in name for bad in DANGER_KEYWORDS)
        ]
        rect = ensure_game_foreground()
        scanned = scan_clickable_texts(cur_img, rect)

        # 优先挑尚未学进 screen_graph.json 且未标记为死胡同的按钮
        learned_here = graph.get(cur_screen, {})
        candidate_name = None
        candidate_xy = None

        for btn in known_btns:
            if btn not in learned_here and (cur_screen, btn) not in tried_deadends:
                candidate_name = btn
                break

        if candidate_name is None:
            for item in scanned:
                txt = item["text"]
                if txt not in learned_here and (cur_screen, txt) not in tried_deadends:
                    candidate_name = txt
                    candidate_xy = item["abs_xy"]
                    break

        if candidate_name is None:
            print(f"[{cur_screen}] 周围安全按钮已全部试探完毕，暂无未知按钮。")
            break

        print(f"[{cur_screen}] 决定试探点击：[{candidate_name}]")
        if candidate_xy is None:
            click_region(candidate_name)
        else:
            click_abs_human(candidate_xy, rect)

        time.sleep(0.9)
        next_screen, next_img = wait_stable_screen(features)

        if next_screen is None:
            record_question(
                q_type="陌生界面",
                question=f"在 [{cur_screen}] 点击 [{candidate_name}] 后进入未知画面，请告诉我它叫什么、该点哪里返回。",
                img=next_img,
                extra={"from_screen": cur_screen, "clicked_btn": candidate_name},
            )
            break
        elif next_screen == cur_screen:
            tried_deadends.add((cur_screen, candidate_name))
            record_question(
                q_type="无效点击",
                question=f"在 [{cur_screen}] 点击 [{candidate_name}] 后界面没有跳转，可能不是可点击按钮或仅弹出小提示。",
                img=next_img,
                extra={"screen": cur_screen, "clicked_btn": candidate_name},
            )
        else:
            graph.setdefault(cur_screen, {})[candidate_name] = next_screen
            save_graph(graph)
            print(f"  [学会新关联！] {cur_screen} --(点 {candidate_name})--> {next_screen}")


def explore_interactive() -> None:
    """半自动引导探索：你指定点哪个按钮，棋魂负责点、看结果、记图谱与存疑"""
    features = load_features(FEATURES_PATH)
    if not features:
        raise FileNotFoundError("请先运行 Learn.py 生成 screen_features.json。")

    regions = load_regions()
    graph = load_graph()

    print("=== 棋魂：引导探索模式 ===")
    print(f"已标记按钮：{list(regions.keys())}\n")

    while True:
        cur_screen, cur_img = wait_stable_screen(features)
        print(f"\n[当前所在界面]：【{cur_screen}】")
        if cur_screen is None:
            record_question("陌生界面", "当前画面不认识，已保存现场截图，请命名后移入 samples/。", cur_img)
            if input("按回车重试识别，输入 q 退出: ").strip().lower() == "q":
                break
            continue

        print(f"[本界面已学会跳转]：{graph.get(cur_screen, {})}")
        btn = input("请指定要试探点击的按钮名称（直接回车退出）: ").strip()
        if not btn:
            break
        if btn not in regions:
            print(f"  [提示] regions.json 中没有 [{btn}]，请先用 Mark.py 标定。")
            continue

        click_region(btn)
        time.sleep(0.8)
        next_screen, next_img = wait_stable_screen(features)

        if next_screen is None:
            record_question(
                "陌生界面",
                f"在 [{cur_screen}] 点击 [{btn}] 后遇到陌生画面，请命名并补标返回按钮。",
                next_img,
                extra={"from_screen": cur_screen, "clicked_btn": btn},
            )
            continue

        graph.setdefault(cur_screen, {})[btn] = next_screen
        save_graph(graph)
        print(f"  [学会了！] {cur_screen} --(点 {btn})--> {next_screen}（已存入 {GRAPH_PATH}）")


def find_path(graph: dict, start: str, target: str) -> Optional[List[str]]:
    """根据学会的 screen_graph.json，用 BFS 规划最短点击路径"""
    if start == target:
        return []
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        cur, path = queue.popleft()
        for btn, nxt in graph.get(cur, {}).items():
            if nxt == target:
                return path + [btn]
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [btn]))
    return None


def navigate_sequence(targets: List[str]) -> None:
    """按顺序自主巡航多个目标界面"""
    features = load_features(FEATURES_PATH)
    graph = load_graph()

    for target in targets:
        print(f"\n=== 新目标：前往【{target}】 ===")
        for _ in range(5):
            cur, cur_img = wait_stable_screen(features)
            if cur == target:
                print(f"已到达【{target}】！")
                break
            if cur is None:
                record_question("陌生界面", f"前往 [{target}] 途中遇到陌生画面，已中止。", cur_img)
                return
            path = find_path(graph, cur, target)
            if not path:
                print(f"[警告] 关系图谱里还没有从【{cur}】到【{target}】的路径。")
                return
            print(f"[自主规划] 当前在【{cur}】，路径={path} -> 点击 [{path[0]}]")
            click_region(path[0])
            time.sleep(0.8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="棋魂 - 自主探索、疑问收集与导航")
    parser.add_argument("--auto", action="store_true", help="开启全自动有边界探索（遇到不懂的自动存进 unknown/）")
    parser.add_argument("--steps", type=int, default=8, help="自动探索最大步数（默认 8 步）")
    parser.add_argument("--tour", nargs="+", help="按顺序巡航多个界面，如: --tour 菜单 大厅 模式选择 大厅")
    args = parser.parse_args()

    if args.auto:
        explore_auto(max_steps=args.steps)
    elif args.tour:
        navigate_sequence(args.tour)
    else:
        explore_interactive()