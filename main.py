import os
import sys
import argparse

# Make analysis/src importable when running from repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import setup_logger, ensure_dir


PROJ_ROOT = os.path.abspath(THIS_DIR)

def _abs_under(base_dir: str, path_str: str) -> str:
    """Resolve path to absolute and group outputs_*/processed_* under analysis parents."""
    if not path_str:
        return os.path.join(PROJ_ROOT, base_dir)
    if os.path.isabs(path_str):
        return path_str
    bn = os.path.basename(path_str.rstrip(os.sep))
    if bn.startswith("outputs_") and base_dir.endswith(os.path.join("analysis", "outputs")):
        return os.path.join(PROJ_ROOT, "outputs", bn) if base_dir == os.path.join("analysis", "outputs") else os.path.join(PROJ_ROOT, base_dir, bn)
    if bn.startswith("processed_") and base_dir.endswith(os.path.join("analysis", "processed")):
        return os.path.join(PROJ_ROOT, "processed", bn) if base_dir == os.path.join("analysis", "processed") else os.path.join(PROJ_ROOT, base_dir, bn)
    return os.path.join(PROJ_ROOT, path_str)


logger = setup_logger("main")


def main():
    parser = argparse.ArgumentParser(description="PRD Mobility Change Pattern: Cross-Period Spatio-Temporal Model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("preprocess", help="Build time series and dynamic graphs from raw CSVs (2021 vs 2024)")
    p_prep.add_argument("--data_dir", type=str, default=os.path.join("analysis", "data"))
    p_prep.add_argument("--out_dir", type=str, default=os.path.join("analysis", "processed"))
    p_prep.add_argument("--year_2021_path", type=str, default=None)
    p_prep.add_argument("--year_2024_path", type=str, default=None)
    p_prep.add_argument("--labels_path", type=str, default=None)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--config", type=str, default=os.path.join("analysis", "configs", "params.yaml"))
    p_train.add_argument("--processed_dir", type=str, default=os.path.join("analysis", "processed"))
    p_train.add_argument("--outputs_dir", type=str, default=os.path.join("analysis", "outputs"))

    p_eval = sub.add_parser("evaluate", help="Evaluate and visualize results")
    p_eval.add_argument("--processed_dir", type=str, default=os.path.join("analysis", "processed"))
    p_eval.add_argument("--outputs_dir", type=str, default=os.path.join("analysis", "outputs"))
    p_eval.add_argument("--best_model", type=str, default=os.path.join("analysis", "outputs", "best_model.pt"))

    args = parser.parse_args()

    if args.cmd == "preprocess":
        from preprocess import run_preprocess
        data_dir = _abs_under(os.path.join("analysis", "data"), args.data_dir)
        out_dir = _abs_under(os.path.join("analysis", "processed"), args.out_dir)
        logger.info(f"preprocess: data_dir={data_dir}")
        logger.info(f"preprocess: processed_dir={out_dir}")
        # year paths and labels may be relative to analysis/; resolve to absolute if given
        def _norm_opt(p):
            if p is None:
                return None
            return p if os.path.isabs(p) else os.path.join(PROJ_ROOT, p.replace("analysis/", "", 1)) if p.startswith("analysis/") else os.path.join(PROJ_ROOT, p)
        run_preprocess(
            data_dir,
            out_dir,
            year_2021_path=_norm_opt(args.year_2021_path),
            year_2024_path=_norm_opt(args.year_2024_path),
            labels_path=_norm_opt(args.labels_path),
        )
    elif args.cmd == "train":
        from train import run_train
        processed_dir = _abs_under(os.path.join("analysis", "processed"), args.processed_dir)
        outputs_dir = _abs_under(os.path.join("analysis", "outputs"), args.outputs_dir)
        ensure_dir(outputs_dir)
        logger.info(f"train: processed_dir={processed_dir}")
        logger.info(f"train: outputs_dir={outputs_dir}")
        run_train(args.config, processed_dir, outputs_dir)
    elif args.cmd == "evaluate":
        from evaluate import run_evaluate
        processed_dir = _abs_under(os.path.join("analysis", "processed"), args.processed_dir)
        outputs_dir = _abs_under(os.path.join("analysis", "outputs"), args.outputs_dir)
        best_model = args.best_model
        if not os.path.isabs(best_model):
            best_model = os.path.join(outputs_dir, os.path.basename(best_model))
        logger.info(f"evaluate: processed_dir={processed_dir}")
        logger.info(f"evaluate: outputs_dir={outputs_dir}")
        logger.info(f"evaluate: best_model={best_model}")
        run_evaluate(processed_dir, outputs_dir, best_model)


if __name__ == "__main__":
    main()
