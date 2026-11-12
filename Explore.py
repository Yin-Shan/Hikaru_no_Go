"""
棋魂 - 自主探索与关系建图模块：Explore.py
====================================================================
职责（第3步：教棋魂自己试着点击，发现“点哪个按钮会去哪个界面”）：
1. 利用 Learn.py 的视觉能力认出“我现在在哪个界面”；
2. 利用 Mouse.py 平滑随机点击某个按钮，等转场动画结束后，再看一眼“我现在到了哪个界面”；
3. 自动把学到的关联关系记录到 screen_graph.json（全程无需手写跳转规则）；
4. 学会关系网后，支持 --goto 参数自主规划最短路径并导航。
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Dict, List, Optional

from Learn import FEATURES_PATH, identify_screen, load_features
from Mouse import click_region, ensure_game_foreground, load_regions
from Window import grab_screen

GRAPH_PATH = "screen_graph.json"


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


def wait_stable_screen(features: dict, timeout: float = 6.0, interval: float = 0.4) -> Optional[str]:
    """等待转场动画结束，直到连续识别到稳定界面"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rect = ensure_game_foreground()
        img = grab_screen(rect)
        screen_name, _ = identify_screen(img, features)
        if screen_name is not None:
            return screen_name
        print("  [动画过渡中] 等待画面稳定...")
        time.sleep(interval)
    return None


def explore_interactive() -> None:
    """
    交互式学习关联性：
    程序先认出当前界面 -> 你告诉它试着点哪个按钮 -> 它自己点完看结果并记住关联。
    """
    features = load_features(FEATURES_PATH)
    if not features:
        raise FileNotFoundError("请先运行 Learn.py 生成 screen_features.json。")

    regions = load_regions()
    graph = load_graph()

    print("=== 棋魂：自主探索界面关联性 ===")
    print(f"已标记的可点击按钮：{list(regions.keys())}")
    print("提示：输入按钮名称让棋魂试着点一次并学习跳转关系；直接回车结束。\n")

    while True:
        cur_screen = wait_stable_screen(features)
        print(f"\n[当前所在界面]：【{cur_screen}】")
        if cur_screen is None:
            print("未能认出当前界面，请手动切回已知界面后按回车重试。")
            if input("按回车继续，输入 q 退出: ").strip().lower() == "q":
                break
            continue

        learned_here = graph.get(cur_screen, {})
        print(f"[本界面已学会的跳转]：{learned_here}")

        btn = input("请指定一个要试探点击的按钮名称（直接回车退出）: ").strip()
        if not btn:
            break
        if btn not in regions:
            print(f"  [提示] regions.json 中没有 [{btn}]，请检查拼写或先用 Mark.py 标定。")
            continue

        # 执行平滑点击并等待动画结束
        click_region(btn)
        time.sleep(0.8)
        next_screen = wait_stable_screen(features)

        if next_screen is None:
            print("  [结果] 点击后进入了一个陌生画面，暂未记录关联（可先截图加入 samples/ 学习）。")
            continue

        # 自动记录关联：cur_screen --(btn)--> next_screen
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
    """按顺序自主走完一串目标界面（例如：菜单 -> 大厅 -> 模式选择 -> 大厅）"""
    features = load_features(FEATURES_PATH)
    graph = load_graph()

    for target in targets:
        print(f"\n=== 新目标：前往【{target}】 ===")
        for _ in range(5):
            cur = wait_stable_screen(features)
            if cur == target:
                print(f"已到达【{target}】！")
                break
            if cur is None:
                print("[警告] 画面识别失败，中止导航。")
                return
            path = find_path(graph, cur, target)
            if not path:
                print(f"[警告] 关系图谱里还没有从【{cur}】到【{target}】的路径，请先用探索模式学习。")
                return
            print(f"[自主规划] 当前在【{cur}】，路径={path} -> 点击 [{path[0]}]")
            click_region(path[0])
            time.sleep(0.8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="棋魂 - 探索界面关联与自主导航")
    parser.add_argument(
        "--tour",
        nargs="+",
        help="测试按顺序自主巡航多个界面，如: --tour 菜单 大厅 模式选择 大厅",
    )
    args = parser.parse_args()

    if args.tour:
        navigate_sequence(args.tour)
    else:
        explore_interactive()