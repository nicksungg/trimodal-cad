# Trimodal CAD

Multimodal CAD code generation from images and point clouds.

This repository accompanies the paper draft in `new.tex` and explores a tri-modal pipeline for generating executable CadQuery programs from:

- multi-view rendered images
- 3D point clouds
- text prompts / CAD code context

## Overview

The project investigates prompt-level multimodal fusion for CAD program generation. Image features and point cloud features are encoded by pretrained backbones, projected into the language model embedding space, and prepended to the textual context before autoregressive decoding.

The paper draft describes:

- a frozen DINOv2 image encoder with learned projection layers
- a pretrained point cloud encoder with projection into LLM space
- CAD-Evolve as the main training corpus
- evaluation on DeepCAD, ABC-Simple, ABC-Complex, MCB, and Objaverse
- three adaptation regimes:
  - linear-only tuning
  - adapter + LLM fine-tuning
  - adapter + LLM + reinforcement learning

## Goals

The main goal is to improve CAD code generation by using explicit 3D geometry alongside 2D visual evidence, especially for shapes with occlusion, holes, and rear-facing features that are difficult to infer from images alone.

## Contents

- `new.tex` — paper draft for the multimodal CAD project
- `summary_placeholder.png` — overview figure used in the draft
- `cad_recode_benchmark/` — benchmark materials and scripts related to CAD-Recode-style evaluation
- `evaluation_data/` — evaluation dataset preparation materials
- `training_data/` — training data preparation materials

## Benchmark Summary

The draft evaluates systems with two primary metrics:

- **STEP Feasibility**: percentage of generated programs that execute and export successfully
- **IoU**: geometric overlap between generated and reference shapes

Datasets are stratified by shape complexity where applicable, with easy/medium/hard splits based on face count.

## Current Status

The paper draft currently includes a reported CAD-Recode baseline and placeholders for:

- CAD-Coder image-only baseline
- linear-only multimodal tuning
- multimodal adapter + LLM fine-tuning
- multimodal adapter + LLM + RL fine-tuning

## Citation

If you use this project or paper draft, please cite the final publication once available.
