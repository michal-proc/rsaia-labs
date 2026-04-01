#!/usr/bin/env python3
"""
Custom Hyperspectral BSQ Viewer (dark theme)
--------------------------------------------
Layout follows the original viewer:
- top toolbar with actions + status
- two plots side by side (RGB + spectral signature)
- bottom matplotlib navigation toolbar
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit("Missing dependency: spectral. Install with `pip install spectral`.")


DATA_DIR = Path(__file__).parent / "data"
ALT_DATA_DIR = DATA_DIR / "images"
FALLBACK_RGB = (30, 20, 10)


def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr")) if directory.exists() else []


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    values = meta.get("wavelength")
    if not values:
        return None
    try:
        return np.array([float(v) for v in values], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def get_rgb_bands(meta: dict, nbands: int) -> tuple[int, int, int]:
    default_bands = meta.get("default bands")
    if default_bands and len(default_bands) >= 3:
        try:
            bands = tuple(int(float(v)) - 1 for v in default_bands[:3])
        except ValueError:
            bands = FALLBACK_RGB
    else:
        bands = FALLBACK_RGB
    return tuple(min(max(v, 0), nbands - 1) for v in bands)


def get_ignore_value(meta: dict) -> float | None:
    raw = meta.get("data ignore value")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def load_image(hdr_path: Path):
    return envi.open(str(hdr_path))


def read_rgb(img, r: int, g: int, b: int, ignore_value: float | None) -> np.ndarray:
    rgb = img.read_bands([r, g, b]).astype(np.float32)

    if ignore_value is not None:
        rgb[rgb == ignore_value] = np.nan
    rgb[rgb < 0] = np.nan

    for c in range(3):
        channel = rgb[:, :, c]
        if not np.isfinite(channel).any():
            rgb[:, :, c] = 0.0
            continue
        low, high = np.nanpercentile(channel, [2, 98])
        rgb[:, :, c] = np.clip((channel - low) / max(high - low, 1e-6), 0, 1)

    return np.nan_to_num(rgb, nan=0.0)


def read_spectrum(img, row: int, col: int, ignore_value: float | None) -> np.ndarray:
    spectrum = img.read_pixel(row, col).astype(np.float64)
    if ignore_value is not None:
        spectrum[spectrum == ignore_value] = np.nan
    spectrum[spectrum < 0] = np.nan
    return spectrum


class DarkNavigationToolbar(NavigationToolbar2Tk):
    toolitems = [
        item
        for item in NavigationToolbar2Tk.toolitems
        if item[0] in ("Subplots", "Save")
    ]

    def __init__(self, canvas, window, bg: str, fg: str):
        self._bg = bg
        self._fg = fg
        super().__init__(canvas, window, pack_toolbar=False)
        self._apply_dark_style()

    def _apply_dark_style(self):
        self.config(bg=self._bg)
        self.update_idletasks()

        def style_widget(widget):
            cls = widget.winfo_class()
            if cls == "Button":
                widget.configure(
                    bg=self._bg,
                    fg=self._fg,
                    activebackground="#2b3440",
                    activeforeground="#ffffff",
                    relief=tk.FLAT,
                    bd=0,
                    highlightthickness=0,
                    highlightbackground=self._bg,
                    padx=4,
                    pady=2,
                )
            elif cls == "Label":
                widget.configure(bg=self._bg, fg=self._fg)
            elif cls == "Entry":
                widget.configure(
                    bg="#151b24",
                    fg=self._fg,
                    insertbackground=self._fg,
                    highlightbackground="#364153",
                    relief=tk.FLAT,
                    bd=0,
                )
            else:
                try:
                    widget.configure(bg=self._bg, highlightthickness=0, bd=0)
                except tk.TclError:
                    pass

            for child in widget.winfo_children():
                style_widget(child)

        style_widget(self)


class HyperspectralViewer:
    BG = "#0f1318"
    PANEL = "#171d26"
    PANEL_ALT = "#10151c"
    TEXT = "#e8edf4"
    TEXT_MUTED = "#9eaabd"
    BTN_BG = "#2a3444"
    BTN_ACTIVE = "#3b4b63"
    BTN_BORDER = "#4f6586"
    ACCENT = "#5ec2f3"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Custom Hyperspectral Viewer")
        self.root.geometry("1320x760")
        self.root.configure(bg=self.BG)

        self.img = None
        self.current_file: Path | None = None
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None = None
        self.rgb_display: np.ndarray | None = None
        self.spectrum: np.ndarray | None = None
        self.pixel_pos: tuple[int, int] | None = None

        self._build_ui()
        self._auto_load()

    def _build_ui(self):
        bar = tk.Frame(self.root, bg=self.PANEL, bd=0, padx=6, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        self._make_action_button(bar, "Open File...", self._open_file).pack(side=tk.LEFT, padx=4)
        self._make_action_button(bar, "Export CSV...", self._export_csv).pack(side=tk.LEFT, padx=4)
        self._make_action_button(bar, "Export PNG...", self._export_signature_png).pack(
            side=tk.LEFT, padx=4
        )

        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(
            bar,
            textvariable=self.status_var,
            anchor=tk.W,
            fg=self.TEXT_MUTED,
            bg=self.PANEL,
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=12)

        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(14, 6.5), facecolor=self.PANEL_ALT)
        self.ax_rgb = self.fig.add_subplot(1, 2, 1, facecolor="#121a25")
        self.ax_spec = self.fig.add_subplot(1, 2, 2, facecolor="#121a25")
        self._style_axes()
        self.fig.tight_layout(pad=2.4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_row = tk.Frame(main, bg=self.PANEL)
        toolbar_row.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = DarkNavigationToolbar(self.canvas, toolbar_row, self.PANEL, self.TEXT_MUTED)
        self.toolbar.pack(side=tk.LEFT, fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._refresh_plots()

    def _make_action_button(self, parent, text: str, command):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            relief=tk.FLAT,
            bd=0,
            bg=self.BTN_BG,
            fg=self.TEXT,
            activebackground=self.BTN_ACTIVE,
            activeforeground="#ffffff",
            font=("Segoe UI Semibold", 10),
            padx=12,
            pady=7,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=self.BTN_BORDER,
            highlightcolor=self.ACCENT,
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg=self.BTN_ACTIVE))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=self.BTN_BG))
        return btn

    def _style_axes(self):
        for ax in (self.ax_rgb, self.ax_spec):
            for spine in ax.spines.values():
                spine.set_color("#3b4659")
            ax.tick_params(colors="#b7c2d0")
            ax.title.set_color(self.TEXT)
            ax.xaxis.label.set_color("#d6deea")
            ax.yaxis.label.set_color("#d6deea")

    def _auto_load(self):
        if len(sys.argv) > 1:
            self._load(Path(sys.argv[1]))
            return

        hdrs = find_hdr_files(ALT_DATA_DIR)
        if not hdrs:
            hdrs = find_hdr_files(DATA_DIR)

        if len(hdrs) == 1:
            self._load(hdrs[0])
            return
        if len(hdrs) > 1:
            self.status_var.set("Multiple .hdr files found. Click 'Open file...' to choose one.")
            return

        self.status_var.set("No .hdr found automatically. Click 'Open file...' to import a dataset.")

    def _open_file(self):
        preferred_dir = ALT_DATA_DIR if ALT_DATA_DIR.exists() else DATA_DIR
        path = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=preferred_dir if preferred_dir.exists() else Path.home(),
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load(self, hdr_path: Path):
        self.status_var.set(f"Loading {hdr_path.name} ...")
        self.root.update_idletasks()

        try:
            self.img = load_image(hdr_path)
            self.current_file = hdr_path
            meta = self.img.metadata
            self.wavelengths = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            r, g, b = get_rgb_bands(meta, self.img.nbands)
            self.rgb_display = read_rgb(self.img, r, g, b, self.ignore_value)
            self.spectrum = None
            self.pixel_pos = None
            self._refresh_plots()
            self.status_var.set(
                f"{hdr_path.name} | {self.img.nrows} x {self.img.ncols} x {self.img.nbands} | "
                "Click a pixel to inspect."
            )
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            self.status_var.set("Load failed. Try another .hdr file.")

    def _refresh_plots(self):
        self.ax_rgb.clear()
        self.ax_spec.clear()
        self._style_axes()

        if self.rgb_display is not None:
            self.ax_rgb.imshow(self.rgb_display, interpolation="bilinear", aspect="auto")
            if self.pixel_pos is not None:
                row, col = self.pixel_pos
                self.ax_rgb.plot(col, row, marker="+", color="#ff6b6b", markersize=14, mew=2.4)
        self.ax_rgb.set_title("RGB preview - click a pixel")
        self.ax_rgb.axis("off")

        if self.spectrum is not None and self.pixel_pos is not None:
            row, col = self.pixel_pos
            x = self._spectrum_x()
            xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
            self.ax_spec.plot(x, self.spectrum, color=self.ACCENT, linewidth=1.4)
            self.ax_spec.fill_between(x, self.spectrum, color=self.ACCENT, alpha=0.12)
            self.ax_spec.grid(True, alpha=0.22, color="#90a6c4")
            self.ax_spec.set_title(f"Spectral signature - row {row}, col {col}")
            self.ax_spec.set_xlabel(xlabel)
            self.ax_spec.set_ylabel("Value")
        else:
            self.ax_spec.set_title("Spectral signature")
            self.ax_spec.text(
                0.5,
                0.5,
                "Click a pixel in the RGB image",
                transform=self.ax_spec.transAxes,
                ha="center",
                va="center",
                color=self.TEXT_MUTED,
                fontsize=12,
            )

        self.fig.tight_layout(pad=2.4)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata)
        row = int(event.ydata)
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return

        self.pixel_pos = (row, col)
        self.spectrum = read_spectrum(self.img, row, col, self.ignore_value)
        self._refresh_plots()
        self.status_var.set(f"Pixel ({row}, {col}) selected. Use CSV/PNG export if needed.")

    def _spectrum_x(self) -> np.ndarray:
        if self.spectrum is None:
            return np.array([], dtype=float)
        if self.wavelengths is not None and len(self.wavelengths) == len(self.spectrum):
            return self.wavelengths
        return np.arange(len(self.spectrum))

    def _export_csv(self):
        if self.spectrum is None or self.pixel_pos is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return

        row, col = self.pixel_pos
        x = self._spectrum_x()
        header = "wavelength_nm" if self.wavelengths is not None else "band"
        path = filedialog.asksaveasfilename(
            title="Save spectrum as CSV",
            initialfile=f"signature_r{row}_c{col}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([header, "value"])
            for xi, vi in zip(x, self.spectrum):
                writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])

        self.status_var.set(f"CSV saved: {path}")

    def _export_signature_png(self):
        if self.spectrum is None or self.pixel_pos is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return

        row, col = self.pixel_pos
        x = self._spectrum_x()
        xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
        path = filedialog.asksaveasfilename(
            title="Save spectral signature as PNG",
            initialfile=f"signature_r{row}_c{col}.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        export_fig = Figure(figsize=(9, 5), facecolor=self.PANEL_ALT)
        export_ax = export_fig.add_subplot(1, 1, 1, facecolor="#121a25")
        export_ax.plot(x, self.spectrum, color=self.ACCENT, linewidth=1.6)
        export_ax.fill_between(x, self.spectrum, color=self.ACCENT, alpha=0.12)
        export_ax.grid(True, alpha=0.22, color="#90a6c4")
        export_ax.set_title(f"Spectral signature - row {row}, col {col}", color=self.TEXT)
        export_ax.set_xlabel(xlabel, color="#d6deea")
        export_ax.set_ylabel("Value", color="#d6deea")
        export_ax.tick_params(colors="#b7c2d0")
        for spine in export_ax.spines.values():
            spine.set_color("#3b4659")
        export_fig.tight_layout(pad=1.8)
        export_fig.savefig(path, dpi=220, facecolor=export_fig.get_facecolor())

        self.status_var.set(f"PNG saved: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()
