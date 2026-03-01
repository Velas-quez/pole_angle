#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Le uma linha de um parquet EVTAC, reconstrui a imagem do sensor e "
            "renderiza original, downsample (2x/4x/8x) e cropped."
        )
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/train/preprocessed/evtac.parquet"),
        help="Caminho para o arquivo parquet EVTAC.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Indice da linha a ser lida (aceita indice negativo).",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=127,
        help="Valor do fundo usado na reconstrucao da imagem.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=2,
        help="Padding em pixels ao redor da regiao ativa durante o crop.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Arquivo de saida da figura. Default: outputs/evtac_row_<idx>_renders.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre a figura em uma janela interativa alem de salvar.",
    )
    parser.add_argument(
        "--grading-gamma",
        type=float,
        default=0.35,
        help="Gamma do color grading (menor = eventos mais destacados).",
    )
    parser.add_argument(
        "--dynamic-read",
        action="store_true",
        help="Se ativado, percorre o parquet linha a linha e gera animacao do sensor.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help=(
            "Amostragem e playback em Hz no modo dinamico. "
            "Ex.: 1.0 => 1 frame por segundo pelo timestamp do parquet."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximo de frames amostrados no modo dinamico (default: sem limite).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Saida da animacao dinamica. Default: outputs/evtac_dynamic.gif",
    )
    return parser.parse_args()


def resolve_row_index(row_index: int, n_rows: int) -> int:
    if n_rows <= 0:
        raise ValueError("O parquet nao possui linhas.")

    resolved = row_index if row_index >= 0 else n_rows + row_index
    if resolved < 0 or resolved >= n_rows:
        raise IndexError(
            f"row-index {row_index} fora do intervalo valido [0, {n_rows - 1}]"
        )
    return resolved


def reconstruct_sensor_image(row: pd.Series, background_value: int) -> np.ndarray:
    height = int(row["height"])
    width = int(row["width"])
    image = np.full(
        (height, width), fill_value=np.uint8(np.clip(background_value, 0, 255))
    )

    indices = np.asarray(row["indices"], dtype=np.int64).ravel()
    values = np.asarray(row["values"], dtype=np.uint8).ravel()
    size = min(indices.size, values.size)
    if size == 0:
        return image

    indices = indices[:size]
    values = values[:size]
    valid = (indices >= 0) & (indices < image.size)
    image.ravel()[indices[valid]] = values[valid]
    return image


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image.copy()

    h, w = image.shape
    h_fit = h - (h % factor)
    w_fit = w - (w % factor)
    if h_fit == 0 or w_fit == 0:
        return image.copy()

    img = image[:h_fit, :w_fit].astype(np.float32)
    reduced = img.reshape(h_fit // factor, factor, w_fit // factor, factor).mean(
        axis=(1, 3)
    )
    return np.clip(np.rint(reduced), 0, 255).astype(np.uint8)


def crop_active_region(
    image: np.ndarray, background_value: int, padding: int = 0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bg = np.uint8(np.clip(background_value, 0, 255))
    active = image != bg
    if not np.any(active):
        h, w = image.shape
        return image.copy(), (0, 0, w, h)

    ys, xs = np.where(active)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(image.shape[0] - 1, int(ys.max()) + padding)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(image.shape[1] - 1, int(xs.max()) + padding)
    return image[y0 : y1 + 1, x0 : x1 + 1], (x0, y0, x1 + 1, y1 + 1)


def apply_color_grading(
    image: np.ndarray, background_value: int, gamma: float
) -> np.ndarray:
    bg = float(np.clip(background_value, 0, 255))
    img = image.astype(np.float32)
    dev = img - bg
    max_dev = max(bg, 255.0 - bg, 1.0)
    strength = np.clip(np.abs(dev) / max_dev, 0.0, 1.0)
    strength = np.power(strength, max(gamma, 1e-3))

    rgb = np.full((image.shape[0], image.shape[1], 3), 0.10, dtype=np.float32)

    pos = dev > 0
    neg = dev < 0

    # Eventos positivos: amarelo/laranja.
    rgb[pos, 0] = 1.00
    rgb[pos, 1] = 0.35 + 0.65 * strength[pos]
    rgb[pos, 2] = 0.05

    # Eventos negativos: ciano/azul.
    rgb[neg, 0] = 0.05
    rgb[neg, 1] = 0.45 + 0.55 * strength[neg]
    rgb[neg, 2] = 1.00

    return np.clip(rgb, 0.0, 1.0)


def render_all_views(
    views: list[tuple[str, np.ndarray]],
    output_path: Path,
    row_info: str,
    show: bool,
    background_value: int,
    grading_gamma: float,
) -> None:
    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, views):
        h, w = img.shape
        color_img = apply_color_grading(img, background_value, grading_gamma)
        ax.imshow(color_img, interpolation="nearest")
        ax.set_title(f"{name}\n{w}x{h} px")
        ax.axis("off")

    fig.suptitle(row_info, fontsize=10)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def compute_views_for_row(
    row: pd.Series, background_value: int, crop_padding: int
) -> tuple[list[tuple[str, np.ndarray]], tuple[int, int, int, int]]:
    original = reconstruct_sensor_image(row, background_value)
    ds2 = downsample_mean(original, 2)
    ds4 = downsample_mean(original, 4)
    ds8 = downsample_mean(original, 8)
    ds16 = downsample_mean(original, 16)
    cropped, crop_bbox = crop_active_region(
        original, background_value, padding=max(crop_padding, 0)
    )

    views = [
        ("Original", original),
        ("Downsample 2x", ds2),
        ("Downsample 4x", ds4),
        ("Downsample 8x", ds8),
        ("Downsample 16x", ds16),
        ("Cropped", cropped),
    ]
    return views, crop_bbox


def _as_float_timestamp(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Timestamp invalido para amostragem temporal: {value!r}") from exc


def iter_evtac_rows_dynamic(
    parquet_path: Path, fps: float, max_frames: int | None = None
) -> tuple[int, list[tuple[int, pd.Series, float]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    if fps <= 0:
        raise ValueError("--fps deve ser maior que zero no modo dinamico.")
    dt = 1.0 / fps

    rows: list[tuple[int, pd.Series, float]] = []
    if pq is not None:
        table = pq.ParquetFile(parquet_path)
        selected = 0
        global_idx = 0
        next_target_ts: float | None = None
        for batch in table.iter_batches(
            batch_size=1024,
            columns=[
                "timestamp",
                "bag_name",
                "height",
                "width",
                "indices",
                "values",
                "nnz",
            ],
        ):
            data = batch.to_pydict()
            batch_len = len(data["timestamp"])
            for i in range(batch_len):
                row_dict = {k: v[i] for k, v in data.items()}
                ts = _as_float_timestamp(row_dict["timestamp"])
                if next_target_ts is None or ts >= next_target_ts:
                    rows.append((global_idx, pd.Series(row_dict), ts))
                    selected += 1
                    if max_frames is not None and selected >= max_frames:
                        break
                    if next_target_ts is None:
                        next_target_ts = ts + dt
                    else:
                        while ts >= next_target_ts:
                            next_target_ts += dt
                global_idx += 1
            if max_frames is not None and selected >= max_frames:
                break
        return table.metadata.num_rows, rows

    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    next_target_ts: float | None = None
    for idx, row in df.iterrows():
        ts = _as_float_timestamp(row["timestamp"])
        if next_target_ts is None or ts >= next_target_ts:
            rows.append((int(idx), row, ts))
            if max_frames is not None and len(rows) >= max_frames:
                break
            if next_target_ts is None:
                next_target_ts = ts + dt
            else:
                while ts >= next_target_ts:
                    next_target_ts += dt
    return n_total, rows


def generate_dynamic_animation(
    parquet_path: Path,
    output_path: Path,
    background_value: int,
    crop_padding: int,
    grading_gamma: float,
    fps: float,
    max_frames: int | None,
) -> None:
    n_total, rows = iter_evtac_rows_dynamic(parquet_path, fps=fps, max_frames=max_frames)
    if not rows:
        raise ValueError("Nao ha linhas para animacao.")

    first_views, _ = compute_views_for_row(rows[0][1], background_value, crop_padding)
    n_panels = len(first_views)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    artists = []
    for ax, (name, img) in zip(axes, first_views):
        h, w = img.shape
        color_img = apply_color_grading(img, background_value, grading_gamma)
        artist = ax.imshow(color_img, interpolation="nearest")
        ax.set_title(f"{name}\n{w}x{h} px")
        ax.axis("off")
        artists.append((artist, ax, name))

    frames: list[Image.Image] = []
    n_rows = len(rows)
    for i, (row_idx, row, ts) in enumerate(rows):
        views, crop_bbox = compute_views_for_row(row, background_value, crop_padding)
        header = (
            f"frame={i + 1}/{n_rows} (total parquet={n_total}) | "
            f"row_idx={row_idx} | bag={row.get('bag_name', 'n/a')} | "
            f"timestamp={ts:.6f} | crop_bbox={crop_bbox}"
        )
        fig.suptitle(header, fontsize=10)

        for (artist, ax, name), (_, img) in zip(artists, views):
            h, w = img.shape
            color_img = apply_color_grading(img, background_value, grading_gamma)
            artist.set_data(color_img)
            ax.set_title(f"{name}\n{w}x{h} px")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = Image.fromarray(rgba[:, :, :3], mode="RGB")
        frames.append(frame)

    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(fps, 0.1))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Animacao salva em: {output_path}")
    print(f"- frames: {len(frames)}")
    print(f"- fps: {fps}")
    if len(rows) >= 2:
        span = rows[-1][2] - rows[0][2]
        print(f"- time span sampled: {span:.3f} s")
    print(f"- total rows parquet: {n_total}")


def main() -> None:
    args = parse_args()
    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {args.parquet_path}")

    if args.dynamic_read:
        video_path = args.video_path
        if video_path is None:
            video_path = Path("outputs") / "evtac_dynamic.gif"
        generate_dynamic_animation(
            parquet_path=args.parquet_path,
            output_path=video_path,
            background_value=args.background_value,
            crop_padding=args.crop_padding,
            grading_gamma=args.grading_gamma,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        return

    df = pd.read_parquet(args.parquet_path)
    idx = resolve_row_index(args.row_index, len(df))
    row = df.iloc[idx]
    views, crop_bbox = compute_views_for_row(row, args.background_value, args.crop_padding)

    output_path = args.output_path
    if output_path is None:
        output_path = Path("outputs") / f"evtac_row_{idx}_renders.png"

    row_info = (
        f"row={idx} | bag={row.get('bag_name', 'n/a')} | "
        f"timestamp={row.get('timestamp', 'n/a')} | crop_bbox={crop_bbox}"
    )
    render_all_views(
        views,
        output_path,
        row_info=row_info,
        show=args.show,
        background_value=args.background_value,
        grading_gamma=args.grading_gamma,
    )

    print(f"Render salvo em: {output_path}")
    for name, img in views:
        h, w = img.shape
        print(f"- {name}: {w}x{h} px")


if __name__ == "__main__":
    main()
