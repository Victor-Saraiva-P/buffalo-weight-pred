from __future__ import annotations


def calculate_hu_moments(pixels: list[tuple[int, int]], area: int) -> tuple[float, float]:
    m00 = float(area)
    m10 = sum(x + 0.5 for x, _ in pixels)
    m01 = sum(y + 0.5 for _, y in pixels)
    cx = m10 / m00
    cy = m01 / m00

    def central_moment(p: int, q: int) -> float:
        return sum(((x + 0.5) - cx) ** p * ((y + 0.5) - cy) ** q for x, y in pixels)

    def normalized_moment(p: int, q: int) -> float:
        return central_moment(p, q) / (m00 ** (1 + (p + q) / 2))

    eta20 = normalized_moment(2, 0)
    eta02 = normalized_moment(0, 2)
    eta11 = normalized_moment(1, 1)
    return eta20 + eta02, (eta20 - eta02) ** 2 + 4 * eta11**2
