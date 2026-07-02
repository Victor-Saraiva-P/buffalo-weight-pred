from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_hu_moment_1(ctx: FeatureContext) -> float:
    eta20 = ctx.moments_data["eta20"]
    eta02 = ctx.moments_data["eta02"]
    return eta20 + eta02


def calculate_hu_moment_2(ctx: FeatureContext) -> float:
    eta20 = ctx.moments_data["eta20"]
    eta02 = ctx.moments_data["eta02"]
    eta11 = ctx.moments_data["eta11"]
    return (eta20 - eta02) ** 2 + 4 * eta11**2
