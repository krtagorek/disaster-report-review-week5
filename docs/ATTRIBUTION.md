# Attribution and asset scope

- **Source data.** NOAA National Centers for Environmental Information and
  National Weather Service, Storm Events Database, 2024 event details, snapshot
  dated July 28, 2026 and originally accessed October 30, 2026. The exact URL,
  checksum and source citation are in [the manifest](../data/source-manifest.json).
  NOAA asks users to cite the dataset and documents source and quality limits.
  Retain event IDs and credited sources where applicable.
  [NOAA dataset metadata and use constraints](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00510/html)
- **Model.** Qwen Team, Qwen3-1.7B-Base, revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, Apache License 2.0. Local adaptation
  used LoRA and native MLX BF16 fusion. The local merge retained the upstream
  license and attribution. This package carries manifests and evaluation
  evidence for those artifacts.
  [Pinned upstream model license](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/LICENSE)
- **Project code and assets.** This academic submission contains project code,
  original project charts and interface captures, split metadata and development
  predictions. The project has no separately declared code license. The Qwen
  license applies to Qwen assets. Dependencies retain their respective licenses.
  NOAA attribution does not grant rights to any separately credited third-party
  material.

The package excludes raw NOAA files, base/adapter/merged weights, runtime review
databases, environment files, caches, the reserved-final manifest, cross-split
duplicate edges and the original Git history.
