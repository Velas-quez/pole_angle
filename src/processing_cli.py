from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from processing.config import get_model_strategy, load_model_map

DEFAULT_EVTAC_NAME = "evtac.parquet"
DEFAULT_MARKERS_NAME = "markers.parquet"
DEFAULT_OPTI_NAME = "optitrack.parquet"


def _resolve_path(input_path: Path, value: str | None, default_name: str) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute() and input_path.is_dir():
            return input_path / path
        return path

    if input_path.is_dir():
        return input_path / default_name

    raise ValueError(
        "--input-path precisa ser um diretório ou você deve informar --evtac/--markers/--opti"
    )


def _print_config(model_map: Dict[str, str]) -> None:
    print("Model -> Processor mapping:")
    for model, processor in sorted(model_map.items()):
        print(f"- {model}: {processor}")

    strategies_dir = Path(__file__).parent / "processing" / "strategies"
    if strategies_dir.exists():
        print("\nAvailable processors:")
        for path in sorted(strategies_dir.glob("*.py")):
            print(f"- {path.stem}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Central processing entrypoint",
        add_help=True,
    )
    parser.add_argument(
        "--list", action="store_true", help="List models and processors"
    )
    parser.add_argument("--model", help="Model name to resolve processor")
    parser.add_argument(
        "--input-path", help="Input path or folder required by processor"
    )
    parser.add_argument("--evtac", help=f"Default: {DEFAULT_EVTAC_NAME}")
    parser.add_argument("--markers", help=f"Default: {DEFAULT_MARKERS_NAME}")
    parser.add_argument("--opti", help=f"Default: {DEFAULT_OPTI_NAME}")
    parser.add_argument("--quad-width", type=float, default=0.25)
    parser.add_argument("--quad-height", type=float, default=0.25)
    parser.add_argument(
        "--output-path",
        default="final_training_data.parquet",
        help="Parquet final (se relativo e --input-path for diretório, salva dentro dele)",
    )
    return parser


def main() -> None:
    model_map = load_model_map()
    parser = _build_parser()
    args = parser.parse_args()

    if args.list:
        _print_config(model_map)
        return

    if not args.model:
        parser.error("You must pass --model (or use --list).")

    if args.model not in model_map:
        parser.error(f"Unknown model '{args.model}'. Use --list to see options.")

    if not args.input_path:
        parser.error("You must pass --input-path.")

    input_path = Path(args.input_path)
    evtac_path = _resolve_path(input_path, args.evtac, DEFAULT_EVTAC_NAME)
    markers_path = _resolve_path(input_path, args.markers, DEFAULT_MARKERS_NAME)
    opti_path = _resolve_path(input_path, args.opti, DEFAULT_OPTI_NAME)

    output_path = Path(args.output_path)
    if not output_path.is_absolute() and input_path.is_dir():
        output_path = input_path / output_path

    strategy = get_model_strategy(
        args.model,
        quad_width=float(args.quad_width),
        quad_height=float(args.quad_height),
    )
    strategy.transform(
        evtac_path=str(evtac_path),
        markers_path=str(markers_path),
        opti_path=str(opti_path),
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()
