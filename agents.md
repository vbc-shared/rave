# RAVE Agents Guide

Instructions for AI agents working on this codebase.

## Project Overview

RAVE (Radiology Vision Engine) is a medical image processing pipeline that converts DICOM and NIfTI files into ML-ready formats. The package is importable as `rve`.

**Core pipeline:** Input (CSV paths) → Processing (modality-specific) → Export (LZ4/HEVC/Torch)

## Architecture

```
vision_engine/
├── cli/main.py              # CLI entry point (vision-engine, rve commands)
├── core/
│   ├── config.py            # YAML config + CLI override handling
│   ├── pipeline.py          # VisionPipeline orchestrator
│   ├── data_structures.py   # SeriesInfo, ProcessedSeries, ExportResult
│   └── exceptions.py        # Custom exception hierarchy
├── input/
│   └── series_paths_loader.py   # CSV/path loading
├── processing/
│   ├── base_processor.py        # Abstract base
│   ├── ct_processor.py          # CT (Hounsfield Units)
│   ├── mr_processor.py          # MRI (signal intensity)
│   ├── xray_processor.py        # 2D X-ray
│   └── mammogram_processor.py   # 2D mammography
├── utils/
│   ├── data_loader.py           # Load tarballs/NIfTI
│   └── windowing_utils.py       # Anatomical windowing
└── *_exporter.py            # LZ4, HEVC video, HEVC image, Torch exporters

rve/                         # Public API package
configs/                     # YAML configurations per modality/anatomy
```

## Key Data Structures

- **SeriesInfo**: Metadata for a DICOM series (accession, series_uid, modality, anatomy)
- **ProcessedSeries**: Numpy volumes/slices ready for export, with processing metadata
- **ExportResult**: Export outcome (path, size, success status)

## Development Guidelines

### Running Tests

```bash
pytest
pytest -x  # stop on first failure
```

### Linting

```bash
ruff check .
ruff check --fix .
```

### Adding a New Modality

1. Create `vision_engine/processing/{modality}_processor.py`
2. Inherit from `BaseProcessor` in `base_processor.py`
3. Register in `get_processor_class()` in `vision_engine/core/pipeline.py`
4. Add YAML config in `configs/{modality}_{anatomy}.yaml`

### Adding a New Exporter

1. Create `vision_engine/{format}_exporter.py`
2. Inherit from `BaseExporter` in `base_exporter.py`
3. Register in `get_exporter_class()` in `vision_engine/core/pipeline.py`
4. Add exporter config in `configs/exporters/`

## Configuration System

All configs are YAML-based. Required fields:
- `modality`: CT, MR, XR, MG
- `anatomy`: anatomical region
- `processing`: modality-specific settings
- `exporter_config`: path to exporter YAML (optional)

CLI flags override YAML values at runtime.

## Important Patterns

1. **Thread safety**: Environment variables `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1` and `OMP_NUM_THREADS=1` are set at import time
2. **Modality normalization**: XR/XRAY, MG/MAMMO/MAMMOGRAPHY, MR/MRI are all valid
3. **Mask auto-detection**: Segmentation masks are auto-detected and use nearest-neighbor resampling + LZ4 export
4. **Metadata preservation**: All exports include JSON metadata with processing details

## Common Tasks

| Task | Location |
|------|----------|
| Add windowing preset | `vision_engine/utils/windowing_utils.py` |
| Modify CLI options | `vision_engine/cli/main.py` |
| Change default processing | `configs/*.yaml` |
| Update public API | `rve/__init__.py` |

## Testing Approach

- Test with `--debug --debug-limit N` to process limited samples
- Use `--dry-run` to validate configuration without processing
- Check `mapping.csv` output for processing results
