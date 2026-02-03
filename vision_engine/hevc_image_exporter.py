"""
HEVC Image exporter for Vision Engine - fast 10-bit intra encoding per image.
Uses system ffmpeg with libx265 (Main10) to encode each image as a single-frame MP4.
"""

import os
import io
import json
import logging
import tarfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .core.data_structures import SeriesInfo, ProcessedSeries, ExportResult
from .core.exceptions import ExportError
from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)


@dataclass
class HEVCImageConfig:
    """Configuration for HEVC image compression"""

    crf: int  # Quality (lower is better, typical 12-24)
    preset: str  # x265 preset: ultrafast..placebo
    pix_fmt: str  # Pixel format, e.g., yuv420p10le or yuv444p10le
    container: str  # 'mp4' or 'mkv'


class HEVCImageExporter(BaseExporter):
    """Export 2D series (Mammography, X-ray) as individual HEVC 10-bit files."""

    def __init__(self, config):
        super().__init__(config)
        hevc_cfg = self.exporter_config.get("hevc_image", {})
        self.hevc_cfg = HEVCImageConfig(
            crf=int(hevc_cfg.get("crf", 18)),
            preset=str(hevc_cfg.get("preset", "medium")),
            pix_fmt=str(hevc_cfg.get("pix_fmt", "yuv420p10le")),
            container=str(hevc_cfg.get("container", "mp4")),
        )
        logger.info(
            f"Initialized HEVC image exporter (crf={self.hevc_cfg.crf}, preset={self.hevc_cfg.preset}, "
            f"pix_fmt={self.hevc_cfg.pix_fmt}, container={self.hevc_cfg.container})"
        )

    def export(
        self, processed_series: ProcessedSeries, output_dir: Path
    ) -> ExportResult:
        series_info = processed_series.series_info
        # Support any 2D slice modality (e.g., MG, XR)
        if processed_series.numpy_slices is None:
            raise ExportError(
                "HEVCImageExporter supports only 2D slice exports (e.g., MG, XR)"
            )

        folder_base = self._derive_series_basename(series_info)
        num_images = len(processed_series.numpy_slices)
        is_multi_image = num_images > 1

        # Get per_image_shapes for InstanceNumber lookup (if available)
        per_image_shapes = processed_series.processing_metadata.get("per_image_shapes", [])

        total_size_mb = 0.0
        count = 0
        output_dirs_created = []

        try:
            for i, slice_array in enumerate(processed_series.numpy_slices):
                # Determine output directory for this image
                if is_multi_image:
                    # Multi-image series: create separate directory per image
                    # Use InstanceNumber if available, otherwise use index
                    instance_num = None
                    if i < len(per_image_shapes):
                        instance_num = per_image_shapes[i].get("instance_number")
                    if instance_num is None:
                        instance_num = i + 1  # 1-based fallback
                    
                    img_folder_name = f"{folder_base}_{instance_num:04d}.0.tar"
                else:
                    # Single-image series: use original naming
                    img_folder_name = f"{folder_base}.0.tar"

                img_output_dir = Path(output_dir) / img_folder_name
                img_output_dir.mkdir(parents=True, exist_ok=True)
                output_dirs_created.append(img_output_dir)

                # Use standardized filename for loader compatibility
                out_path = img_output_dir / f"volume.{self.hevc_cfg.container}"

                if out_path.exists() and not getattr(self.config, "overwrite", False):
                    logger.info(f"Skipping existing file: {out_path}")
                    total_size_mb += out_path.stat().st_size / (1024 * 1024)
                    count += 1
                    continue

                # Write a temporary 16-bit PNG as input to ffmpeg
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_png = Path(tmpdir) / "frame.png"
                    png_bytes = self._array_to_png16(slice_array)
                    with open(tmp_png, "wb") as f:
                        f.write(png_bytes)

                    # ffmpeg command: single frame, intra-only, main10
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(tmp_png),
                        "-c:v",
                        "libx265",
                        "-pix_fmt",
                        self.hevc_cfg.pix_fmt,
                        "-x265-params",
                        "keyint=1:scenecut=0:open-gop=0:repeat-headers=1",
                        "-preset",
                        self.hevc_cfg.preset,
                        "-crf",
                        str(self.hevc_cfg.crf),
                        "-frames:v",
                        "1",
                        str(out_path),
                    ]
                    logger.debug(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode != 0:
                        raise ExportError(
                            f"ffmpeg failed: {result.stderr.decode('utf-8', 'ignore')}"
                        )

                total_size_mb += out_path.stat().st_size / (1024 * 1024)
                count += 1

                # Save metadata JSON in each image's directory
                meta_path = img_output_dir / "metadata.json"
                # Create per-image metadata
                img_metadata = self._create_per_image_metadata(
                    processed_series, i, per_image_shapes
                )
                with open(meta_path, "w") as f:
                    f.write(img_metadata)

            # For single-image, output_path is the single directory
            # For multi-image, use the base output_dir (parent of all image dirs)
            final_output_path = str(output_dirs_created[0]) if not is_multi_image else str(output_dir)

            return ExportResult(
                series_info=series_info,
                output_path=final_output_path,
                file_size_mb=total_size_mb,
                slice_count=count,
                success=True,
            )
        except Exception as e:
            raise ExportError(f"Failed to export HEVC images: {e}")

    def _array_to_png16(self, array: np.ndarray) -> bytes:
        # Ensure 16-bit PNG input for ffmpeg
        if array.dtype == np.uint16:
            img = Image.fromarray(array, mode="I;16")
        elif array.dtype == np.int16:
            arr = (array.astype(np.int32) + 32768).astype(np.uint16)
            img = Image.fromarray(arr, mode="I;16")
        else:
            # Normalize to 16-bit
            min_val = float(np.min(array))
            max_val = float(np.max(array))
            if max_val > min_val:
                norm = ((array - min_val) / (max_val - min_val) * 65535.0).astype(
                    np.uint16
                )
            else:
                norm = np.zeros_like(array, dtype=np.uint16)
            img = Image.fromarray(norm, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=3)
        return buf.getvalue()

    def _create_metadata_json(self, processed_series: ProcessedSeries) -> str:
        metadata = {
            "series_info": {
                "accession": processed_series.series_info.accession,
                "series_number": processed_series.series_info.series_number,
                "series_uid": processed_series.series_info.series_uid,
                "series_description": processed_series.series_info.series_description,
                "modality": processed_series.series_info.modality,
                "slice_count": processed_series.get_slice_count(),
            },
            "processing_metadata": processed_series.processing_metadata,
            "export_info": {
                "format": "HEVC",
                "profile": "Main10",
                "pix_fmt": self.hevc_cfg.pix_fmt,
                "crf": self.hevc_cfg.crf,
                "preset": self.hevc_cfg.preset,
                "container": self.hevc_cfg.container,
                "exporter_version": "1.0",
            },
        }
        return json.dumps(metadata, indent=2)

    def _create_per_image_metadata(
        self, processed_series: ProcessedSeries, image_idx: int, per_image_shapes: List
    ) -> str:
        """Create metadata JSON for a single image in a multi-image series."""
        # Get image-specific shape info
        image_shape_info = {}
        if image_idx < len(per_image_shapes):
            image_shape_info = per_image_shapes[image_idx]

        metadata = {
            "series_info": {
                "accession": processed_series.series_info.accession,
                "series_number": processed_series.series_info.series_number,
                "series_uid": processed_series.series_info.series_uid,
                "series_description": processed_series.series_info.series_description,
                "modality": processed_series.series_info.modality,
                "total_images_in_series": len(processed_series.numpy_slices),
                "image_index": image_idx,
            },
            "image_info": image_shape_info,
            "processing_metadata": {
                k: v for k, v in processed_series.processing_metadata.items()
                if k != "per_image_shapes"  # Exclude full list, we have image_info
            },
            "export_info": {
                "format": "HEVC",
                "profile": "Main10",
                "pix_fmt": self.hevc_cfg.pix_fmt,
                "crf": self.hevc_cfg.crf,
                "preset": self.hevc_cfg.preset,
                "container": self.hevc_cfg.container,
                "exporter_version": "2.0",  # Bumped for 1:1 directory structure
            },
        }
        return json.dumps(metadata, indent=2)

    def _derive_series_basename(self, series_info: SeriesInfo) -> str:
        """Derive a human-meaningful base filename for outputs.
        Preference: series_description -> directory/file name -> series_uid -> accession/series_number.
        Avoids generic 'unknown_*' names."""
        # 1) Series description
        desc = (getattr(series_info, "series_description", "") or "").strip()
        if desc:
            return self._sanitize_name(desc)
        # 2) Directory or file name from path
        try:
            path = series_info.get_dicom_path()
            base = os.path.basename(path.rstrip("/"))
            if os.path.isfile(path):
                base = os.path.splitext(base)[0]
            if base:
                return self._sanitize_name(base)
        except Exception:
            pass
        # 3) Series UID
        uid = (getattr(series_info, "series_uid", "") or "").strip()
        if uid:
            return self._sanitize_name(uid)
        # 4) Accession + series number (last resort)
        acc = (getattr(series_info, "accession", "") or "").strip()
        if acc and not acc.startswith("unknown_"):
            return self._sanitize_name(f"{acc}_{series_info.series_number}")
        return self._sanitize_name(f"series_{series_info.series_number}")

    def _sanitize_name(self, name: str) -> str:
        allowed = []
        for ch in name:
            if ch.isalnum() or ch in ["-", "_", "."]:
                allowed.append(ch)
            else:
                allowed.append("_")
        # Collapse repeated underscores
        sanitized = "".join(allowed)
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_") or "series"
