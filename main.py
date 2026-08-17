"""CLI entry point for the animal image classifier.

Commands:
    python main.py train      -- train the transfer-learning classifier
    python main.py evaluate   -- per-class report + top-1 precision + confusion matrix
    python main.py predict    -- classify an image/folder, emit structured JSON
"""
import argparse

from config import (
    MODEL_NAME, EPOCHS, BATCH_SIZE, LR, FREEZE_BACKBONE, SEED,
    RAW_IMAGE_DIR, MODELS_DIR, OUTPUTS_DIR,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="image-recognition",
                                description="Animal image classifier (capstone starter).")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train the transfer-learning classifier.")
    tr.add_argument("--model", default=MODEL_NAME, help="resnet18 | mobilenet_v3_small")
    tr.add_argument("--epochs", type=int, default=EPOCHS)
    tr.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    tr.add_argument("--lr", type=float, default=LR)
    tr.add_argument("--freeze-backbone", action="store_true", default=FREEZE_BACKBONE)
    tr.add_argument("--raw-image-dir", default=str(RAW_IMAGE_DIR))
    tr.add_argument("--seed", type=int, default=SEED)

    ev = sub.add_parser("evaluate", help="Evaluate the trained model on the validation split.")
    ev.add_argument("--checkpoint", default=str(MODELS_DIR / "best.pt"))
    ev.add_argument("--raw-image-dir", default=str(RAW_IMAGE_DIR))
    ev.add_argument("--output-dir", default=str(OUTPUTS_DIR))

    pr = sub.add_parser("predict", help="Classify image(s) and emit structured JSON.")
    pr.add_argument("--checkpoint", default=str(MODELS_DIR / "best.pt"))
    pr.add_argument("--image", default=None, help="Path to a single image.")
    pr.add_argument("--dir", dest="dir_", default=None, help="Path to a folder of images.")
    pr.add_argument("--topk", type=int, default=3)

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.command == "train":
        from train import train_all
        train_all(
            model_name=args.model, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, freeze_backbone=args.freeze_backbone,
            raw_image_dir=args.raw_image_dir, seed=args.seed,
        )
    elif args.command == "evaluate":
        from evaluate import evaluate
        evaluate(checkpoint=args.checkpoint, raw_image_dir=args.raw_image_dir,
                 output_dir=args.output_dir)
    elif args.command == "predict":
        source = args.image or args.dir_
        if not source:
            print("predict requires --image <path> or --dir <path>", file=__import__("sys").stderr)
            return 2
        from predict import predict_main
        predict_main(checkpoint=args.checkpoint, source=source, topk=args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
