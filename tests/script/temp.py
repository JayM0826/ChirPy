from PIL import Image, ImageDraw, ImageFont
import os

# ====== 设置 ======
gif_list = [
    ("achiral.gif", "achiral_combined.png"),
    # ("chiral.gif", "chiral_combined.png")
]
frames_per_row = 5
frame_font_size = 48
title_font_size = 72

# ==== 字体加载 ====
font_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
]

font = None
for path in font_candidates:
    if os.path.exists(path):
        font = ImageFont.truetype(path, frame_font_size)
        print(f"✅ 使用字体: {path}")
        font_path_used = path
        break

if font is None:
    print("⚠️ 所有候选字体加载失败，使用默认字体")
    font = ImageFont.load_default()
    title_font = font
else:
    title_font = ImageFont.truetype(font_path_used, title_font_size)

# ===== GIF 处理函数 =====
def process_gif(gif_path, output_image, frames_per_row):
    # 从文件名获取标题
    if "chiral" in gif_path.lower():
        title_text = "Achiral Path"
    elif "achiral" in gif_path.lower():
        title_text = "Achiral Path"
    else:
        title_text = "Trajectory"

    with Image.open(gif_path) as im:
        frames = []
        for frame_index in range(min(im.n_frames, 19)):
            im.seek(frame_index)
            frame = im.convert("RGB").copy()

            # 添加每帧标签
            draw = ImageDraw.Draw(frame)
            label = f"Frame {frame_index + 1}"
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            x = (frame.width - label_width) // 2
            y = 10
            draw.text((x, y), label, fill="black", font=font)
            frames.append(frame)

    # 获取帧尺寸和行列
    w, h = frames[0].size
    total = len(frames)
    rows = (total + frames_per_row - 1) // frames_per_row

    # 准备输出图尺寸（包含标题）
    title_height = title_font_size + 20
    combined_width = frames_per_row * w
    combined_height = rows * h + title_height

    combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    draw_combined = ImageDraw.Draw(combined)

    # 添加标题
    title_bbox = draw_combined.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (combined_width - title_width) // 2
    title_y = (title_height - title_font_size) // 2
    draw_combined.text((title_x, title_y), title_text, fill="black", font=title_font)

    # 贴图帧
    for idx, frame in enumerate(frames):
        x = (idx % frames_per_row) * w
        y = (idx // frames_per_row) * h + title_height
        combined.paste(frame, (x, y))

    combined.save(output_image, dpi=(300, 300))
    print(f"✅ Saved {output_image} with title '{title_text}'")

# ===== 执行全部 =====
for gif_path, output_image in gif_list:
    process_gif(gif_path, output_image, frames_per_row)
