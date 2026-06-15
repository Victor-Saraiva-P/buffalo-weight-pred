from __future__ import annotations


def calculate_perimeter(mask: list[list[bool]], pixels: list[tuple[int, int]]) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    perimeter = 0
    for x, y in pixels:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or not mask[ny][nx]:
                perimeter += 1
    return perimeter
