"""
global_id에 해당하는 모든 트랙렛의 썸네일을 하나의 이미지(sheet)로 합성하여 저장합니다.

사용법:
    python scripts/export_global_id_sheet.py <global_id> [--filtered-dir data/filtered]
                                                          [--out output/sheets]
                                                          [--thumb-size 80]
                                                          [--cols 10]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


THUMB_W = 80   # 썸네일 너비 (px)
THUMB_H = 160  # 썸네일 높이 (px, 인물 이미지 비율 고려)
LABEL_H = 14   # 파일명 레이블 높이
HEADER_H = 20  # 트랙렛 헤더 높이
PAD = 4        # 썸네일 간격
BG_COLOR = (240, 240, 240)
HEADER_BG = (60, 90, 140)
HEADER_FG = (255, 255, 255)
LABEL_FG = (50, 50, 50)


def load_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def find_tracklets(filtered_dir: Path, global_id: int) -> list[tuple[Path, dict]]:
    """global_id가 일치하는 (트랙렛 디렉토리, metadata) 목록 반환"""
    results = []
    for meta_path in sorted(filtered_dir.glob("*/track_*/metadata.json")):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("global_id") == global_id:
            results.append((meta_path.parent, meta))
    return results


def make_thumb(img_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), BG_COLOR)
    x = (THUMB_W - img.width) // 2
    y = (THUMB_H - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def render_tracklet_block(tdir: Path, meta: dict, cols: int, font_label, font_header) -> Image.Image:
    images = sorted(tdir.glob("*.jpg"))
    if not images:
        return None

    cam = meta.get("camera_id", "?")
    slot = meta.get("time_slot", "?")
    track = meta.get("track_id", "?")
    header_text = f"cam{cam}_t{slot} / track_{track:04d}  ({len(images)}장)"

    rows = (len(images) + cols - 1) // cols
    cell_w = THUMB_W + PAD
    cell_h = THUMB_H + LABEL_H + PAD

    block_w = cols * cell_w + PAD
    block_h = HEADER_H + rows * cell_h + PAD
    block = Image.new("RGB", (block_w, block_h), BG_COLOR)
    draw = ImageDraw.Draw(block)

    # 헤더
    draw.rectangle([0, 0, block_w, HEADER_H], fill=HEADER_BG)
    draw.text((PAD, (HEADER_H - font_header.size) // 2), header_text,
              fill=HEADER_FG, font=font_header)

    for idx, img_path in enumerate(images):
        row, col = divmod(idx, cols)
        x = PAD + col * cell_w
        y = HEADER_H + PAD + row * cell_h

        thumb = make_thumb(img_path)
        block.paste(thumb, (x, y))

        label = img_path.stem  # e.g. frame_032772
        draw.text((x, y + THUMB_H + 1), label, fill=LABEL_FG, font=font_label)

    return block


MAX_HEIGHT = 60000  # JPEG 최대 허용 높이


def stack_blocks(blocks: list[Image.Image], sheet_w: int) -> Image.Image:
    """블록들을 세로로 이어 붙여 하나의 sheet 생성"""
    total_h = sum(b.height + PAD for b in blocks) + PAD
    sheet = Image.new("RGB", (sheet_w, total_h), BG_COLOR)
    y = PAD
    for block in blocks:
        sheet.paste(block, (PAD, y))
        y += block.height + PAD
    return sheet


def split_and_save(blocks: list[Image.Image], sheet_w: int,
                   out_dir: Path, stem: str) -> list[Path]:
    """MAX_HEIGHT를 초과하면 자동으로 여러 파일로 분할 저장"""
    pages: list[list[Image.Image]] = []
    current: list[Image.Image] = []
    current_h = PAD

    for block in blocks:
        needed = block.height + PAD
        if current and current_h + needed > MAX_HEIGHT:
            pages.append(current)
            current = []
            current_h = PAD
        current.append(block)
        current_h += needed

    if current:
        pages.append(current)

    saved = []
    for i, page_blocks in enumerate(pages):
        sheet = stack_blocks(page_blocks, sheet_w)
        suffix = f"_part{i + 1:02d}" if len(pages) > 1 else ""
        out_path = out_dir / f"{stem}{suffix}.jpg"
        sheet.save(out_path, "JPEG", quality=92)
        print(f"[OK] {out_path.name}  ({sheet.width}x{sheet.height}px)")
        saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Global ID 썸네일 시트 생성")
    parser.add_argument("global_id", type=int, help="조회할 Global ID")
    parser.add_argument("--filtered-dir", default="data/filtered",
                        help="filtered 데이터 루트 경로 (기본: data/filtered)")
    parser.add_argument("--out", default="output/sheets",
                        help="출력 디렉토리 (기본: output/sheets)")
    parser.add_argument("--thumb-size", type=int, default=80,
                        help="썸네일 너비 px (기본: 80)")
    parser.add_argument("--cols", type=int, default=10,
                        help="트랙렛당 열 수 (기본: 10)")
    args = parser.parse_args()

    global THUMB_W, THUMB_H
    THUMB_W = args.thumb_size
    THUMB_H = args.thumb_size * 2

    filtered_dir = Path(args.filtered_dir)
    if not filtered_dir.exists():
        sys.exit(f"[ERROR] filtered 디렉토리를 찾을 수 없습니다: {filtered_dir.resolve()}")

    print(f"[INFO] Global ID {args.global_id} 검색 중...")
    tracklets = find_tracklets(filtered_dir, args.global_id)

    if not tracklets:
        sys.exit(f"[ERROR] Global ID {args.global_id}에 해당하는 트랙렛이 없습니다.")

    total_imgs = sum(len(list(tdir.glob("*.jpg"))) for tdir, _ in tracklets)
    print(f"[INFO] 트랙렛 {len(tracklets)}개, 이미지 총 {total_imgs}장 발견")

    font_label = load_font(9)
    font_header = load_font(11)

    blocks = []
    for tdir, meta in tracklets:
        block = render_tracklet_block(tdir, meta, args.cols, font_label, font_header)
        if block:
            blocks.append(block)

    if not blocks:
        sys.exit("[ERROR] 렌더링할 이미지가 없습니다.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_w = max(b.width for b in blocks) + PAD * 2
    stem = f"global_{args.global_id:04d}_sheet"
    saved = split_and_save(blocks, sheet_w, out_dir, stem)
    print(f"[완료] {len(saved)}개 파일 → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
