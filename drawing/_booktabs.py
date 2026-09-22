"""Booktabs-style table drawing: no vertical rules, thick outer lines."""

from __future__ import annotations

from matplotlib.axes import Axes

_TOP = 1.35
_MID = 0.90
_GROUP = 0.45
_BOTTOM = 1.35


def draw_booktabs(
    ax: Axes,
    headers: list[str],
    cells: list[list[str]],
    *,
    col_x: list[float],
    col_align: list[str],
    bold_row: list[bool] | None = None,
    group_after: list[int] | None = None,
    span_col0: list[tuple[int, int, str]] | None = None,
    fontsize: float = 9.5,
) -> None:
    """Draw ``headers`` plus ``cells`` in axes coordinates.

    ``group_after`` lists data-row indices (0-based) after which a thin
    rule is drawn. ``span_col0`` is ``(first, last, label)`` inclusive
    ranges whose task name is centered on the first column.
    """
    n_data = len(cells)
    n_cols = len(headers)
    bold_row = bold_row or [False] * n_data
    group_after = group_after or []
    span_col0 = span_col0 or []

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_axis_off()

    n_total = n_data + 1
    top = 0.97
    bottom = 0.04
    row_h = (top - bottom) / n_total

    def y_line(after_header_rows: float) -> float:
        return top - after_header_rows * row_h

    def y_text(row_from_top: float) -> float:
        return top - (row_from_top + 0.5) * row_h

    def hline(y: float, lw: float) -> None:
        ax.plot(
            [0.015, 0.985],
            [y, y],
            color="black",
            lw=lw,
            solid_capstyle="butt",
            clip_on=False,
            transform=ax.transAxes,
        )

    def put(x: float, y: float, text: str, *, align: str, bold: bool, size: float) -> None:
        ha = "left" if align == "left" else "center"
        ax.text(
            x,
            y,
            text,
            ha=ha,
            va="center",
            fontsize=size,
            fontweight="bold" if bold else "normal",
            fontfamily="serif",
            transform=ax.transAxes,
            clip_on=False,
        )

    hline(y_line(0.0), _TOP)
    for c in range(n_cols):
        put(
            col_x[c],
            y_text(0.0),
            headers[c],
            align=col_align[c],
            bold=True,
            size=fontsize,
        )
    hline(y_line(1.0), _MID)

    spanned = set()
    for first, last, _ in span_col0:
        spanned.update(range(first, last + 1))

    for i, row in enumerate(cells):
        y = y_text(1.0 + i)
        is_bold = bold_row[i]
        for c, text in enumerate(row):
            if c == 0 and i in spanned:
                continue
            put(
                col_x[c],
                y,
                text,
                align=col_align[c],
                bold=is_bold,
                size=fontsize,
            )
        if i in group_after:
            hline(y_line(2.0 + i), _GROUP)

    for first, last, label in span_col0:
        y_mid = 0.5 * (y_text(1.0 + first) + y_text(1.0 + last))
        put(col_x[0], y_mid, label, align=col_align[0], bold=True, size=fontsize)

    hline(y_line(1.0 + n_data), _BOTTOM)
