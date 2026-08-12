# burnseg-xai

Unsupervised burned-area detection in Landsat 8 time-series imagery.

Four study regions: Karipuna, Kayapó, Yanomami (Amazon), and Chapada dos
Guimarães (Cerrado).

## Dataset

Published at
[yingyangtongxue/br_burned_aois_dataset](https://github.com/yingyangtongxue/br_burned_aois_dataset).
Download it into `./data` before running any of the commands below.

You can also pull the Landsat 8 imagery yourself from
[Google Earth Engine](https://earthengine.google.com/) (needs its own API
credentials); see `src/burnseg_xai/acquisition/` for the download pipeline.

Region boundaries and INPE fire-hotspot references used by the pipeline are
in `aoi/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` if you need to override the default paths
(`./data`, `./outputs`) or set Earth Engine credentials.

## Running

Train a single experiment:

```bash
python -m burnseg_xai.pipeline.run_experiment --config configs/config.yaml
```

Leave-one-region-out and cross-biome evaluation:

```bash
python -m burnseg_xai.pipeline.run_loro --config configs/config.yaml
```

Quick smoke test (20 epochs, 500 patches):

```bash
python -m burnseg_xai.pipeline.run_experiment --config configs/config_quick.yaml
```

Run the tests:

```bash
pytest tests/unit tests/integration -v
```

## Status

This repository accompanies a manuscript currently under review. Citation
details will follow once it is published.

## License

[MIT](LICENSE)
