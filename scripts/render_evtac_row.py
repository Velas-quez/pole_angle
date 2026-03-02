#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R


DEFAULT_BG = 127


def parse_downsamples(raw: str) -> list[int]:
    txt = raw.strip()
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]
    if not txt:
        raise ValueError("Lista de downsamples vazia.")

    values = []
    for token in txt.replace(" ", "").split(","):
        if not token:
            continue
        v = int(token)
        if v < 1:
            raise ValueError(f"Downsample invalido: {v} (deve ser >= 1)")
        values.append(v)

    if not values:
        raise ValueError("Nenhum downsample valido informado.")

    # Remove duplicatas preservando ordem
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render EVTAC com downsamples (grid), e pose 3D base+bastao "
            "sincronizada por timestamp."
        )
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/train/preprocessed/evtac.parquet"),
        help="Caminho do parquet EVTAC.",
    )
    parser.add_argument(
        "--markers-path",
        type=Path,
        default=None,
        help="Caminho do parquet de markers. Default: <dir_evtac>/markers.parquet",
    )
    parser.add_argument(
        "--opti-path",
        type=Path,
        default=None,
        help="Caminho do parquet do optitrack. Default: <dir_evtac>/optitrack.parquet",
    )

    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Indice da linha EVTAC no modo estatico (aceita negativo).",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=DEFAULT_BG,
        help="Valor de fundo do sensor na reconstrucao da imagem.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=2,
        help="Padding (px) ao redor da regiao ativa no crop.",
    )
    parser.add_argument(
        "--downsamples",
        type=str,
        default="1,2,4,8,16",
        help="Lista de fatores de downsample. Exemplos: '1,2,4,10' ou '[1,2,4,10]'.",
    )
    parser.add_argument(
        "--render-cropped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Renderiza ou nao o painel de imagem cropped.",
    )
    parser.add_argument(
        "--show-projection-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sobrepoe vetores no EVTAC: verde (projecao XY do bastao) e "
            "laranja (projecao relativa bastao-base)."
        ),
    )
    parser.add_argument(
        "--projection-intensity-multiplier",
        type=float,
        default=4.0,
        help=(
            "Multiplicador de intensidade visual dos vetores de projecao no EVTAC. "
            "Valores maiores deixam os vetores mais longos."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Saida da figura estatica. Default: outputs/evtac_row_<idx>_renders.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre a figura no modo estatico.",
    )

    parser.add_argument(
        "--dynamic-read",
        action="store_true",
        help="Modo dinamico: amostra por timestamp e gera animacao.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help=(
            "Taxa de amostragem (Hz) por timestamp e tambem taxa de playback. "
            "Ex.: 1.0 => 1 frame por segundo real."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximo de frames selecionados no modo dinamico.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Saida da animacao. Default: outputs/evtac_dynamic.gif",
    )

    args = parser.parse_args()
    args.downsamples = parse_downsamples(args.downsamples)
    return args


def resolve_aux_paths(evtac_path: Path, markers_path: Path | None, opti_path: Path | None) -> tuple[Path, Path]:
    base_dir = evtac_path.parent
    markers = markers_path if markers_path is not None else base_dir / "markers.parquet"
    opti = opti_path if opti_path is not None else base_dir / "optitrack.parquet"
    return markers, opti


def resolve_row_index(row_index: int, n_rows: int) -> int:
    if n_rows <= 0:
        raise ValueError("O parquet EVTAC nao possui linhas.")
    idx = row_index if row_index >= 0 else n_rows + row_index
    if idx < 0 or idx >= n_rows:
        raise IndexError(f"row-index {row_index} fora de [0, {n_rows - 1}]")
    return idx


def _as_float_timestamp(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Timestamp invalido: {value!r}") from exc


def reconstruct_sensor_image(row: pd.Series, background_value: int) -> np.ndarray:
    h = int(row["height"])
    w = int(row["width"])
    bg = np.uint8(np.clip(background_value, 0, 255))
    image = np.full((h, w), fill_value=bg, dtype=np.uint8)

    indices = np.asarray(row["indices"], dtype=np.int64).ravel()
    values = np.asarray(row["values"], dtype=np.uint8).ravel()
    n = min(indices.size, values.size)
    if n == 0:
        return image

    indices = indices[:n]
    values = values[:n]
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
    reduced = image[:h_fit, :w_fit].astype(np.float32).reshape(
        h_fit // factor, factor, w_fit // factor, factor
    ).mean(axis=(1, 3))
    return np.clip(np.rint(reduced), 0, 255).astype(np.uint8)


def crop_active_region(image: np.ndarray, background_value: int, padding: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
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


def compute_deviation_range(image: np.ndarray, background_value: int) -> tuple[float, float]:
    dev_min = 0.0
    dev_max = 0.0
    dev = image.astype(np.float32) - float(background_value)
    dev_min = min(dev_min, float(np.min(dev)))
    dev_max = max(dev_max, float(np.max(dev)))
    return dev_min, dev_max


def apply_signed_evtac_colormap(
    image: np.ndarray, background_value: int, dev_min: float, dev_max: float
) -> np.ndarray:
    dev = image.astype(np.float32) - float(background_value)
    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

    pos = dev > 0
    neg = dev < 0

    pos_scale = max(dev_max, 1e-6)
    neg_scale = max(abs(dev_min), 1e-6)

    pos_t = np.clip(dev[pos] / pos_scale, 0.0, 1.0)
    neg_t = np.clip((-dev[neg]) / neg_scale, 0.0, 1.0)

    # Positivo: preto -> laranja
    rgb[pos, 0] = 1.00 * pos_t
    rgb[pos, 1] = 0.55 * pos_t
    rgb[pos, 2] = 0.00

    # Negativo: preto -> azul
    rgb[neg, 0] = 0.00
    rgb[neg, 1] = 0.45 * neg_t
    rgb[neg, 2] = 1.00 * neg_t

    return np.clip(rgb, 0.0, 1.0)


def compute_views_for_row(
    row: pd.Series,
    background_value: int,
    crop_padding: int,
    downsamples: list[int],
    render_cropped: bool,
) -> tuple[list[tuple[str, np.ndarray]], tuple[int, int, int, int]]:
    original = reconstruct_sensor_image(row, background_value)
    cropped, bbox = crop_active_region(original, background_value, max(crop_padding, 0))

    views: list[tuple[str, np.ndarray]] = []
    for factor in downsamples:
        if factor == 1:
            img = original
            label = "Downsample 1x"
        else:
            img = downsample_mean(original, factor)
            label = f"Downsample {factor}x"
        views.append((label, img))

    if render_cropped:
        views.append(("Cropped", cropped))

    return views, bbox


def estimate_pole_vector(markers) -> np.ndarray:
    arr = np.asarray(markers, dtype=np.float64).ravel()
    if arr.size < 6 or arr.size % 3 != 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pts = arr.reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    centered = pts - np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(centered)
    vec = vh[0]
    if vec[2] < 0:
        vec = -vec
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / n


def parse_orientation_quat(orientation) -> np.ndarray:
    if isinstance(orientation, dict):
        q = np.array(
            [
                orientation.get("x", 0.0),
                orientation.get("y", 0.0),
                orientation.get("z", 0.0),
                orientation.get("w", 1.0),
            ],
            dtype=np.float64,
        )
    else:
        arr = np.asarray(orientation, dtype=np.float64).ravel()
        q = arr[:4] if arr.size >= 4 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def parse_position(pos) -> np.ndarray:
    if isinstance(pos, dict):
        p = np.array([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)], dtype=np.float64)
    else:
        arr = np.asarray(pos, dtype=np.float64).ravel()
        p = arr[:3] if arr.size >= 3 else np.zeros(3, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p


def base_pose_from_orientation(
    orientation, pos
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quat = parse_orientation_quat(orientation)
    rot = R.from_quat(quat)
    base_vec = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    base_euler = rot.as_euler("xyz", degrees=True)
    n = np.linalg.norm(base_vec)
    if n < 1e-9:
        base_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        base_vec = base_vec / n
    base_pos = parse_position(pos)
    return base_pos, base_vec, base_euler, quat


def vector_to_roll_pitch_yaw_deg(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    roll = np.degrees(np.arctan2(y, z))
    pitch = np.degrees(np.arctan2(-x, np.sqrt(y * y + z * z) + 1e-9))
    yaw = np.degrees(np.arctan2(y, x + 1e-9))
    return np.array([roll, pitch, yaw], dtype=np.float64)


@dataclass
class PoseState:
    base_pos: np.ndarray
    base_vec: np.ndarray
    base_euler_deg: np.ndarray
    base_quat: np.ndarray
    pole_vec: np.ndarray
    pole_euler_deg: np.ndarray
    base_ts: float | None
    pole_ts: float | None


class PoseLookup:
    def __init__(self, markers_df: pd.DataFrame, opti_df: pd.DataFrame):
        self._marker_groups: dict[str, tuple[np.ndarray, list]] = {}
        self._opti_groups: dict[str, tuple[np.ndarray, list, list]] = {}

        for df in (markers_df, opti_df):
            if "bagfile" in df.columns and "bag_name" not in df.columns:
                df.rename(columns={"bagfile": "bag_name"}, inplace=True)

        if "bag_name" not in markers_df.columns:
            markers_df = markers_df.copy()
            markers_df["bag_name"] = "_default"
        if "bag_name" not in opti_df.columns:
            opti_df = opti_df.copy()
            opti_df["bag_name"] = "_default"

        m = markers_df.sort_values("timestamp")
        for bag, g in m.groupby("bag_name", dropna=False):
            bag_key = str(bag)
            self._marker_groups[bag_key] = (
                g["timestamp"].astype(float).to_numpy(),
                g["markers"].tolist(),
            )

        o = opti_df.sort_values("timestamp")
        if "pos" not in o.columns:
            o = o.copy()
            o["pos"] = [np.zeros(3)] * len(o)

        for bag, g in o.groupby("bag_name", dropna=False):
            bag_key = str(bag)
            self._opti_groups[bag_key] = (
                g["timestamp"].astype(float).to_numpy(),
                g["orientation"].tolist(),
                g["pos"].tolist(),
            )

        self._default_marker_bag = next(iter(self._marker_groups), None)
        self._default_opti_bag = next(iter(self._opti_groups), None)

        self.pos_min = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        self.pos_max = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        all_pos = []
        for _, (_, _, pos_list) in self._opti_groups.items():
            for p in pos_list:
                all_pos.append(parse_position(p))
        if all_pos:
            arr = np.vstack(all_pos)
            self.pos_min = np.min(arr, axis=0)
            self.pos_max = np.max(arr, axis=0)

    @staticmethod
    def _nearest(ts_arr: np.ndarray, target_ts: float) -> int:
        if ts_arr.size == 0:
            return -1
        pos = int(np.searchsorted(ts_arr, target_ts))
        if pos <= 0:
            return 0
        if pos >= ts_arr.size:
            return int(ts_arr.size - 1)
        left, right = pos - 1, pos
        return left if abs(ts_arr[left] - target_ts) <= abs(ts_arr[right] - target_ts) else right

    def _get_group(self, groups: dict, bag_name: str | None, default_bag: str | None):
        if bag_name is not None and bag_name in groups:
            return groups[bag_name]
        if default_bag is not None and default_bag in groups:
            return groups[default_bag]
        if groups:
            return groups[next(iter(groups))]
        return None

    def lookup(self, bag_name: str | None, timestamp: float) -> PoseState:
        bag_key = str(bag_name) if bag_name is not None else None

        marker_group = self._get_group(self._marker_groups, bag_key, self._default_marker_bag)
        if marker_group is not None:
            ts_arr, markers_list = marker_group
            i_m = self._nearest(ts_arr, timestamp)
            pole_ts = float(ts_arr[i_m]) if i_m >= 0 else None
            pole_vec = estimate_pole_vector(markers_list[i_m]) if i_m >= 0 else np.array([0.0, 0.0, 1.0])
        else:
            pole_ts = None
            pole_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        opti_group = self._get_group(self._opti_groups, bag_key, self._default_opti_bag)
        if opti_group is not None:
            ts_arr, orient_list, pos_list = opti_group
            i_o = self._nearest(ts_arr, timestamp)
            base_ts = float(ts_arr[i_o]) if i_o >= 0 else None
            if i_o >= 0:
                base_pos, base_vec, base_euler, base_quat = base_pose_from_orientation(
                    orient_list[i_o], pos_list[i_o]
                )
            else:
                base_pos = np.zeros(3, dtype=np.float64)
                base_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                base_euler = np.zeros(3, dtype=np.float64)
                base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            base_ts = None
            base_pos = np.zeros(3, dtype=np.float64)
            base_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            base_euler = np.zeros(3, dtype=np.float64)
            base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        pole_euler = vector_to_roll_pitch_yaw_deg(pole_vec)
        return PoseState(
            base_pos=base_pos,
            base_vec=base_vec,
            base_euler_deg=base_euler,
            base_quat=base_quat,
            pole_vec=pole_vec,
            pole_euler_deg=pole_euler,
            base_ts=base_ts,
            pole_ts=pole_ts,
        )


def select_rows_by_timestamp(evtac_df: pd.DataFrame, fps: float, max_frames: int | None) -> list[tuple[int, pd.Series, float]]:
    if fps <= 0:
        raise ValueError("--fps deve ser maior que zero.")
    dt = 1.0 / fps

    selected: list[tuple[int, pd.Series, float]] = []
    next_target: float | None = None

    for idx, row in evtac_df.iterrows():
        ts = _as_float_timestamp(row["timestamp"])
        if next_target is None or ts >= next_target:
            selected.append((int(idx), row, ts))
            if max_frames is not None and len(selected) >= max_frames:
                break
            if next_target is None:
                next_target = ts + dt
            else:
                while ts >= next_target:
                    next_target += dt
    return selected


def setup_layout(n_views: int, lookup: PoseLookup) -> tuple[plt.Figure, list[plt.Axes], plt.Axes]:
    side = int(np.ceil(np.sqrt(max(n_views, 1))))

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.06)

    left = gs[0, 0].subgridspec(side, side, wspace=0.05, hspace=0.15)
    image_axes = [fig.add_subplot(left[r, c]) for r in range(side) for c in range(side)]

    ax_pose = fig.add_subplot(gs[0, 1], projection="3d")
    ax_pose.set_xlabel("X")
    ax_pose.set_ylabel("Y")
    ax_pose.set_zlabel("Z")

    base_span = lookup.pos_max - lookup.pos_min
    span = max(float(np.max(base_span)), 0.3)
    center = (lookup.pos_min + lookup.pos_max) / 2.0
    radius = span * 0.75 + 0.5

    ax_pose.set_xlim(center[0] - radius, center[0] + radius)
    ax_pose.set_ylim(center[1] - radius, center[1] + radius)
    ax_pose.set_zlim(center[2] - radius, center[2] + radius)

    ax_pose.grid(True, alpha=0.3)
    return fig, image_axes, ax_pose


def update_pose_axis(
    ax_pose: plt.Axes,
    base_line,
    pole_line,
    base_point,
    top_point,
    pose: PoseState,
    base_len: float = 0.40,
    pole_len: float = 0.90,
) -> None:
    base_bottom = pose.base_pos
    base_top = base_bottom + pose.base_vec * base_len
    pole_tip = base_top + pose.pole_vec * pole_len

    base_line.set_data_3d(
        [base_bottom[0], base_top[0]],
        [base_bottom[1], base_top[1]],
        [base_bottom[2], base_top[2]],
    )
    pole_line.set_data_3d(
        [base_top[0], pole_tip[0]],
        [base_top[1], pole_tip[1]],
        [base_top[2], pole_tip[2]],
    )

    base_point.set_data_3d([base_bottom[0]], [base_bottom[1]], [base_bottom[2]])
    top_point.set_data_3d([base_top[0]], [base_top[1]], [base_top[2]])

    br, bp, by = pose.base_euler_deg
    pr, pp, py = pose.pole_euler_deg
    ax_pose.set_title(
        "Base + Bastao (3D)\n"
        f"base_pos=({base_bottom[0]:.3f},{base_bottom[1]:.3f},{base_bottom[2]:.3f}) | "
        f"base r/p/y=({br:.1f},{bp:.1f},{by:.1f}) deg\n"
        f"pole r/p/y=({pr:.1f},{pp:.1f},{py:.1f}) deg | "
        f"ts_base={pose.base_ts if pose.base_ts is not None else 'n/a'} | "
        f"ts_pole={pose.pole_ts if pose.pole_ts is not None else 'n/a'}"
    )


def _safe_unit_xy(v3: np.ndarray) -> tuple[np.ndarray, float]:
    v = np.asarray([v3[0], v3[1]], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(2, dtype=np.float64), 0.0
    return v / n, n


def draw_evtac_projection_vectors(
    ax: plt.Axes, image: np.ndarray, pose: PoseState, intensity_multiplier: float
) -> list:
    h, w = image.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    max_len = max(4.0, 0.35 * min(h, w))

    # EVTAC esta rigidamente preso na base: projetamos no frame da base/sensor.
    base_rot = R.from_quat(pose.base_quat)
    pole_local = base_rot.inv().apply(pose.pole_vec)
    base_local = base_rot.inv().apply(pose.base_vec)

    pole_u, pole_mag = _safe_unit_xy(pole_local)
    base_u, _ = _safe_unit_xy(base_local)

    # Direcao relativa no plano XY: diferenca das projecoes.
    rel_xy = np.asarray(
        [base_local[0] - pole_local[0], base_local[1] - pole_local[1]],
        dtype=np.float64,
    )
    rel_xy_n = float(np.linalg.norm(rel_xy))
    if rel_xy_n < 1e-9:
        rel_u = pole_u if pole_mag > 1e-9 else np.zeros(2, dtype=np.float64)
    else:
        rel_u = rel_xy / rel_xy_n

    # Magnitude relativa: proporcional ao angulo 3D entre bastao e base.
    pole_n = np.linalg.norm(pole_local)
    base_n = np.linalg.norm(base_local)
    if pole_n < 1e-9 or base_n < 1e-9:
        angle = 0.0
    else:
        cosang = float(np.dot(pole_local, base_local) / (pole_n * base_n))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        angle = float(np.arccos(cosang))  # [0, pi]

    mul = max(float(intensity_multiplier), 0.0)
    pole_len = max_len * min(mul * pole_mag, 1.0)
    rel_len = max_len * min(mul * (angle / np.pi), 1.0)

    # y da imagem cresce para baixo, por isso invertido no componente y.
    # Vetores invertidos conforme solicitado.
    u_green, v_green = -pole_u[0] * pole_len, pole_u[1] * pole_len
    u_orange, v_orange = -rel_u[0] * rel_len, rel_u[1] * rel_len

    green = ax.quiver(
        cx,
        cy,
        u_green,
        v_green,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#00ff66",
        width=0.010,
        alpha=0.95,
    )
    orange = ax.quiver(
        cx,
        cy,
        u_orange,
        v_orange,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#ff9900",
        width=0.010,
        alpha=0.95,
    )
    return [green, orange]


def render_static(
    views: list[tuple[str, np.ndarray]],
    pose: PoseState,
    lookup: PoseLookup,
    background_value: int,
    show_projection_vectors: bool,
    projection_intensity_multiplier: float,
    output_path: Path,
    row_info: str,
    show: bool,
) -> None:
    fig, image_axes, ax_pose = setup_layout(len(views), lookup)

    for ax, (name, img) in zip(image_axes, views):
        h, w = img.shape
        dev_min, dev_max = compute_deviation_range(img, background_value=background_value)
        ax.imshow(
            apply_signed_evtac_colormap(
                img,
                background_value=background_value,
                dev_min=dev_min,
                dev_max=dev_max,
            ),
            interpolation="nearest",
        )
        ax.set_title(f"{name}\n{w}x{h} px")
        ax.axis("off")
        if show_projection_vectors:
            draw_evtac_projection_vectors(
                ax, img, pose, intensity_multiplier=projection_intensity_multiplier
            )

    for ax in image_axes[len(views) :]:
        ax.axis("off")

    (base_line,) = ax_pose.plot([0, 0], [0, 0], [0, 0], color="tab:orange", lw=4, label="Base")
    (pole_line,) = ax_pose.plot([0, 0], [0, 0], [0, 0], color="tab:green", lw=4, label="Bastao")
    (base_point,) = ax_pose.plot([0], [0], [0], marker="o", color="tab:red", markersize=6, linestyle="None", label="Posicao Base")
    (top_point,) = ax_pose.plot([0], [0], [0], marker="o", color="tab:blue", markersize=6, linestyle="None", label="Topo Base")
    update_pose_axis(ax_pose, base_line, pole_line, base_point, top_point, pose)
    ax_pose.legend(loc="upper left")

    fig.suptitle(row_info, fontsize=10)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def generate_dynamic_animation(
    evtac_df: pd.DataFrame,
    lookup: PoseLookup,
    output_path: Path,
    background_value: int,
    crop_padding: int,
    fps: float,
    max_frames: int | None,
    downsamples: list[int],
    render_cropped: bool,
    show_projection_vectors: bool,
    projection_intensity_multiplier: float,
) -> None:
    rows = select_rows_by_timestamp(evtac_df, fps=fps, max_frames=max_frames)
    if not rows:
        raise ValueError("Nao ha linhas selecionadas para animacao.")

    _, first_row, first_ts = rows[0]
    first_views, _ = compute_views_for_row(
        first_row,
        background_value,
        crop_padding,
        downsamples=downsamples,
        render_cropped=render_cropped,
    )
    first_pose = lookup.lookup(str(first_row.get("bag_name", "")), first_ts)

    fig, image_axes, ax_pose = setup_layout(len(first_views), lookup)

    img_artists = []
    vector_artists: list[list] = []
    for ax, (name, img) in zip(image_axes, first_views):
        h, w = img.shape
        dev_min, dev_max = compute_deviation_range(img, background_value)
        artist = ax.imshow(
            apply_signed_evtac_colormap(
                img, background_value=background_value, dev_min=dev_min, dev_max=dev_max
            ),
            interpolation="nearest",
        )
        ax.set_title(f"{name}\n{w}x{h} px")
        ax.axis("off")
        img_artists.append((artist, ax, name))
        if show_projection_vectors:
            vector_artists.append(
                draw_evtac_projection_vectors(
                    ax,
                    img,
                    first_pose,
                    intensity_multiplier=projection_intensity_multiplier,
                )
            )
        else:
            vector_artists.append([])

    for ax in image_axes[len(first_views) :]:
        ax.axis("off")

    (base_line,) = ax_pose.plot([0, 0], [0, 0], [0, 0], color="tab:orange", lw=4, label="Base")
    (pole_line,) = ax_pose.plot([0, 0], [0, 0], [0, 0], color="tab:green", lw=4, label="Bastao")
    (base_point,) = ax_pose.plot([0], [0], [0], marker="o", color="tab:red", markersize=6, linestyle="None", label="Posicao Base")
    (top_point,) = ax_pose.plot([0], [0], [0], marker="o", color="tab:blue", markersize=6, linestyle="None", label="Topo Base")
    update_pose_axis(ax_pose, base_line, pole_line, base_point, top_point, first_pose)
    ax_pose.legend(loc="upper left")

    frames: list[Image.Image] = []
    for i, (row_idx, row, ts) in enumerate(rows):
        views, crop_bbox = compute_views_for_row(
            row,
            background_value,
            crop_padding,
            downsamples=downsamples,
            render_cropped=render_cropped,
        )
        pose = lookup.lookup(str(row.get("bag_name", "")), ts)

        for (artist, ax, name), (_, img) in zip(img_artists, views):
            h, w = img.shape
            dev_min, dev_max = compute_deviation_range(img, background_value)
            artist.set_data(
                apply_signed_evtac_colormap(
                    img,
                    background_value=background_value,
                    dev_min=dev_min,
                    dev_max=dev_max,
                )
            )
            ax.set_title(f"{name}\n{w}x{h} px")

        if show_projection_vectors:
            for i_ax, (_, ax, _) in enumerate(img_artists):
                for art in vector_artists[i_ax]:
                    art.remove()
                vector_artists[i_ax] = draw_evtac_projection_vectors(
                    ax,
                    views[i_ax][1],
                    pose,
                    intensity_multiplier=projection_intensity_multiplier,
                )

        update_pose_axis(ax_pose, base_line, pole_line, base_point, top_point, pose)

        fig.suptitle(
            f"frame={i + 1}/{len(rows)} | row_idx={row_idx} | bag={row.get('bag_name', 'n/a')} | "
            f"timestamp={ts:.6f} | crop_bbox={crop_bbox}",
            fontsize=10,
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))

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
    print(f"- frames: {len(rows)}")
    print(f"- fps (sampling/playback): {fps}")
    if len(rows) >= 2:
        print(f"- time span sampled: {rows[-1][2] - rows[0][2]:.3f} s")


def main() -> None:
    args = parse_args()

    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Arquivo EVTAC nao encontrado: {args.parquet_path}")

    markers_path, opti_path = resolve_aux_paths(args.parquet_path, args.markers_path, args.opti_path)
    if not markers_path.exists():
        raise FileNotFoundError(f"Arquivo markers nao encontrado: {markers_path}")
    if not opti_path.exists():
        raise FileNotFoundError(f"Arquivo optitrack nao encontrado: {opti_path}")

    evtac_df = pd.read_parquet(args.parquet_path)
    markers_df = pd.read_parquet(markers_path)
    opti_df = pd.read_parquet(opti_path)
    lookup = PoseLookup(markers_df=markers_df, opti_df=opti_df)

    if args.dynamic_read:
        video_path = args.video_path if args.video_path is not None else Path("outputs") / "evtac_dynamic.gif"
        generate_dynamic_animation(
            evtac_df=evtac_df,
            lookup=lookup,
            output_path=video_path,
            background_value=args.background_value,
            crop_padding=args.crop_padding,
            fps=args.fps,
            max_frames=args.max_frames,
            downsamples=args.downsamples,
            render_cropped=args.render_cropped,
            show_projection_vectors=args.show_projection_vectors,
            projection_intensity_multiplier=args.projection_intensity_multiplier,
        )
        return

    idx = resolve_row_index(args.row_index, len(evtac_df))
    row = evtac_df.iloc[idx]
    ts = _as_float_timestamp(row["timestamp"])
    bag = str(row.get("bag_name", ""))

    views, crop_bbox = compute_views_for_row(
        row,
        args.background_value,
        args.crop_padding,
        downsamples=args.downsamples,
        render_cropped=args.render_cropped,
    )
    pose = lookup.lookup(bag_name=bag, timestamp=ts)

    output_path = args.output_path if args.output_path is not None else Path("outputs") / f"evtac_row_{idx}_renders.png"
    row_info = f"row={idx} | bag={bag} | timestamp={ts:.6f} | crop_bbox={crop_bbox}"

    render_static(
        views=views,
        pose=pose,
        lookup=lookup,
        background_value=args.background_value,
        show_projection_vectors=args.show_projection_vectors,
        projection_intensity_multiplier=args.projection_intensity_multiplier,
        output_path=output_path,
        row_info=row_info,
        show=args.show,
    )

    print(f"Render salvo em: {output_path}")
    for name, img in views:
        h, w = img.shape
        print(f"- {name}: {w}x{h} px")


if __name__ == "__main__":
    main()
