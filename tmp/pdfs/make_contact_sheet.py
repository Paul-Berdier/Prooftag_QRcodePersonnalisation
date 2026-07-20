from pathlib import Path
from PIL import Image, ImageOps

folder = Path(__file__).parent / "rendered_translation"
pages = sorted(folder.glob("page-*.png"))
thumbs = []
for page in pages:
    image = Image.open(page).convert("RGB")
    width = 320
    height = round(image.height * width / image.width)
    image = image.resize((width, height))
    thumbs.append(ImageOps.expand(image, border=2, fill=(100, 100, 100)))

rows = []
for start in range(0, len(thumbs), 3):
    group = thumbs[start : start + 3]
    row_height = max(image.height for image in group)
    row = Image.new(
        "RGB",
        (sum(image.width for image in group) + 12 * (len(group) - 1), row_height),
        "#D9DEE3",
    )
    x = 0
    for image in group:
        row.paste(image, (x, 0))
        x += image.width + 12
    rows.append(row)

sheet = Image.new(
    "RGB",
    (max(row.width for row in rows), sum(row.height for row in rows) + 12 * (len(rows) - 1)),
    "#B9C1C8",
)
y = 0
for row in rows:
    sheet.paste(row, (0, y))
    y += row.height + 12
sheet.save(folder / "contact_sheet.png")
