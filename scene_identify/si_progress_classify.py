#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import CLIPModel, AutoProcessor


class LocalCLIPClassifier:
    """
    CLIP classifier for zero-shot image classification
    """

    def __init__(
        self,
        model_ref: str = "openai/clip-vit-base-patch32",
        cache_dir: str = "./lmodel_cache",
        device: str | None = None,
    ):
        """
        Input:
            model_ref: Model name or local model folder
            cache_dir: Local cache location
            device: "cuda" / "cpu" / None
        """
        self.model_ref = model_ref
        self.cache_dir = cache_dir

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] model_ref={self.model_ref}")
        print(f"[INFO] cache_dir={self.cache_dir}")
        print("[INFO] local_files_only=True")
        print(f"[INFO] device={self.device}")

        # 僅允許本地載入
        self.processor = AutoProcessor.from_pretrained(
            self.model_ref,
            cache_dir=self.cache_dir,
            local_files_only=True,
        )

        self.model = CLIPModel.from_pretrained(
            self.model_ref,
            cache_dir=self.cache_dir,
            local_files_only=True,
        ).to(self.device)

        self.model.eval()

    def classify(
        self,
        image_path: str,
        candidate_labels: List[str],
        text_template: str = "a screenshot of {}",
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        """
        Input:
            image_path: Image path
            candidate_labels: Candidate labels, e.g., ["HOME", "BATTLE", "RESULT"]
            text_template: Template to wrap labels into prompts
            top_k: Only keep top k results, None means all outputs

        Output:
            {
              "image_path": "...",
              "best_label": "...",
              "scores": [
                {"label": "...", "prob": 0.91},
                {"label": "...", "prob": 0.07},
                ...
              ]
            }
        """
        if not candidate_labels:
            raise ValueError("candidate_labels cannot be empty")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        prompts = [text_template.format(label) for label in candidate_labels]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_labels]
            probs = logits_per_image.softmax(dim=1)[0]

        results = []
        for label, prob in zip(candidate_labels, probs.tolist()):
            results.append({
                "label": label,
                "prob": float(prob),
            })

        results.sort(key=lambda x: x["prob"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return {
            "image_path": str(path),
            "best_label": results[0]["label"],
            "scores": results,
        }


def main():
    #
    candidate_labels = [
        "HOME",
        "SORTIE_PREP",
        "BATTLE",
        "RESULT",
        "OTHER",
        "a girl with some equipment on her back"
    ]

    # directly point to a local model folder
    model_ref = "./scene_identify/models/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

    classifier = LocalCLIPClassifier(
        model_ref=model_ref,
        cache_dir="./scene_identify/models",
        device="cpu",
    )

    result = classifier.classify(
        image_path="./kc_simulator/screenshots/drop_2.png",
        candidate_labels=candidate_labels,
        text_template="a game screenshot of {}",
        top_k=5,
    )

    print("\n[OUTPUT]")
    print(f"image_path : {result['image_path']}")
    print(f"best_label : {result['best_label']}")
    print("scores     :")
    for item in result["scores"]:
        print(f"  - {item['label']}: {item['prob']:.4f}")


if __name__ == "__main__":
    main()

# down load model
# Make sure the hf CLI is installed
#powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# Download the model
#hf download openai/clip-vit-base-patch32 --cache-dir ./scene_identify/models