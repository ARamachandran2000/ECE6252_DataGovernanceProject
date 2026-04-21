#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from transformers import AutoConfig, AutoModelForSpeechSeq2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Whisper model architecture to XML"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-base",
        help="Hugging Face model id (default: openai/whisper-base)",
    )
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=Path("outputs") / "whisper_base_architecture.xml",
        help="Path to output XML file",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--with-pretrained-weights",
        action="store_true",
        help="Load pretrained weights instead of config-only init",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional max module depth to export",
    )
    return parser.parse_args()


def local_param_count(module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def total_param_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def trainable_param_count(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def describe_module(module) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key in [
        "in_features",
        "out_features",
        "num_embeddings",
        "embedding_dim",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "normalized_shape",
        "hidden_size",
        "num_attention_heads",
    ]:
        if hasattr(module, key):
            attrs[key] = str(getattr(module, key))
    return attrs


def add_module_xml(parent: ET.Element, name: str, module, depth: int, max_depth: int | None) -> None:
    node = ET.SubElement(
        parent,
        "module",
        {
            "name": name,
            "type": module.__class__.__name__,
            "depth": str(depth),
            "local_params": str(local_param_count(module)),
            "total_params": str(total_param_count(module)),
            "trainable_params": str(trainable_param_count(module)),
        },
    )

    details = describe_module(module)
    if details:
        details_node = ET.SubElement(node, "details")
        for key, value in details.items():
            ET.SubElement(details_node, "attr", {"name": key, "value": value})

    if max_depth is not None and depth >= max_depth:
        return

    for child_name, child in module.named_children():
        add_module_xml(node, child_name, child, depth + 1, max_depth)


def main() -> int:
    args = parse_args()

    args.output_xml.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )

    if args.with_pretrained_weights:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_id,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        load_mode = "from_pretrained"
    else:
        model = AutoModelForSpeechSeq2Seq.from_config(config)
        load_mode = "from_config"

    root = ET.Element(
        "whisper_architecture",
        {
            "model_id": args.model_id,
            "model_class": model.__class__.__name__,
            "load_mode": load_mode,
            "total_params": str(total_param_count(model)),
            "trainable_params": str(trainable_param_count(model)),
        },
    )

    config_node = ET.SubElement(root, "config")
    cfg = config.to_dict()
    for key in sorted(cfg.keys()):
        ET.SubElement(config_node, "item", {"key": str(key), "value": str(cfg[key])})

    modules_node = ET.SubElement(root, "modules")
    add_module_xml(modules_node, "model", model, depth=0, max_depth=args.max_depth)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(args.output_xml, encoding="utf-8", xml_declaration=True)

    print(f"Saved XML to: {args.output_xml.resolve()}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total params: {total_param_count(model):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
