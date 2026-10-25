"""Presentation-only PNG exports with typed table sidecars and an asset manifest."""
from pathlib import Path
import textwrap

from IPython.display import Image, display
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
import numpy as np

from .module01_forecasting import file_digest, write_json


def primitive(value):
    return value.item() if isinstance(value, np.generic) else value


def format_cell(value):
    value = primitive(value)
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Represent missing table results as None")
        return f"{value:.2f}"
    return str(value)


class PublicationAssets:
    def __init__(self, cfg, source_hash):
        self.cfg, self.source_hash, self.assets = cfg, source_hash, []
        self.cfg.graph_folder.mkdir(exist_ok=True)
        self.cfg.example_folder.mkdir(parents=True, exist_ok=True)

    def save(self, fig, kind, section, cell_id, caption, table_data=None):
        number = sum(a["kind"] == kind for a in self.assets) + 1
        filename = f"atsf01-{kind}{number}.png"
        destination = self.cfg.graph_folder / filename
        if kind == "table":
            if fig.texts or fig._suptitle or any(ax.get_title() or ax.texts for ax in fig.axes):
                raise ValueError("Table figures must have no titles, footer or free text")
        fig.savefig(destination, dpi=self.cfg.export_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        asset = {"kind": kind, "number": number, "filename": filename, "path": f"graphs/{filename}",
                 "section": section, "cell_id": cell_id, "caption": caption,
                 "source_result_hash": self.source_hash, "sha256": file_digest(destination),
                 "post": "a01-post.md", "dpi": self.cfg.export_dpi,
                 "font": Path(findfont(FontProperties(family=["sans-serif"]))).name,
                 "figsize": list(fig.get_size_inches())}
        if table_data is not None:
            asset["table_sidecar"] = f"examples/module01/{filename.removesuffix('.png')}.json"
            write_json(self.cfg.root / asset["table_sidecar"], table_data)
            asset["table_sidecar_sha256"] = file_digest(self.cfg.root / asset["table_sidecar"])
            asset["free_text_artists"] = 0
        self.assets.append(asset)
        write_json(self.cfg.example_folder / "assets.json", self.assets)
        display(Image(filename=str(destination)))
        return destination

    def table(self, blocks, section, cell_id, caption):
        """Each block is (headers, typed rows, relative column widths)."""
        prepared, sidecars, line_totals = [], [], []
        for headers, rows, widths in blocks:
            if not all(len(row) == len(headers) for row in rows):
                raise ValueError("Table row width mismatch")
            raw = [[primitive(v) for v in row] for row in rows]
            rendered = [[format_cell(v) for v in row] for row in raw]
            # Text wraps more aggressively in narrow columns, without changing values.
            limits = [max(8, int(self.cfg.table_wrap_chars * width)) for width in widths]
            wrap = lambda row: [textwrap.fill(str(v), limit, break_long_words=False) for v, limit in zip(row, limits)]
            head, values = wrap(headers), [wrap(row) for row in rendered]
            heights = [max(v.count("\n") + 1 for v in row) + 1 for row in [head, *values]]
            total = sum(heights)
            prepared.append((head, values, widths, heights, total))
            line_totals.append(total)
            sidecars.append({"headers": headers, "raw": raw, "display": rendered,
                             "types": [[type(v).__name__ for v in row] for row in raw]})
        height = max(self.cfg.table_min_height, sum(line_totals) * self.cfg.table_line_height)
        fig, axes = plt.subplots(len(blocks), 1, figsize=(self.cfg.img_dim1 * self.cfg.publication_scale, height),
                                 gridspec_kw={"height_ratios": line_totals, "hspace": 0.22}, squeeze=False)
        for ax, (head, values, widths, heights, total) in zip(axes.flat, prepared):
            ax.axis("off")
            table = ax.table(cellText=values, colLabels=head, colWidths=widths,
                             cellLoc="left", bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(self.cfg.table_fontsize)
            for (row, col), cell in table.get_celld().items():
                cell.set_height(heights[row] / total)
                cell.set_edgecolor("white")
                cell.PAD = 0.06
                cell.set_facecolor(self.cfg.table_header_color if row == 0 else self.cfg.table_alternate_color if row % 2 else "white")
                if row == 0:
                    cell.set_text_props(color="white", weight="bold")
        return self.save(fig, "table", section, cell_id, caption, sidecars)
