from __future__ import annotations

import gzip
import math
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote

import numpy as np

INPUT_COMPATIBILITY_VERSION = "optional_colortable_mrb_v1"

def _header(stream):
    first = stream.readline(128)
    if not first.startswith(b"NRRD000"):
        raise ValueError("MRB segmentation/reference must be NRRD.")
    header = {}
    total = len(first)
    while True:
        line = stream.readline(((1024 * 1024) + 1))
        total += len(line)
        if (total > (1024 * 1024)):
            raise ValueError("NRRD header is too large.")
        if (line in (b"\n", b"\r\n")):
            break
        if not line:
            raise ValueError("Truncated NRRD header.")
        text = line.decode("utf-8-sig").strip()
        if (not text or text.startswith("#")):
            continue
        separator = (":=" if (":=" in text) else ":")
        if (separator not in text):
            raise ValueError("Malformed NRRD header field.")
        key, value = text.split(separator, 1)
        header[key.strip()] = value.strip()
    return header

def _sizes(header):
    sizes = tuple(int(v) for v in header["sizes"].split())
    if ((len(sizes) != int(header["dimension"])) or not sizes or (min(sizes) <= 0)):
        raise ValueError("Invalid NRRD dimensions/sizes.")
    return sizes

def _payload(stream, header):
    types = {
        "uchar": "u1",
        "unsigned char": "u1",
        "uint8": "u1",
        "uint8_t": "u1",
        "char": "i1",
        "signed char": "i1",
        "int8": "i1",
        "int8_t": "i1",
        "ushort": "u2",
        "unsigned short": "u2",
        "unsigned short int": "u2",
        "uint16": "u2",
        "uint16_t": "u2",
        "short": "i2",
        "short int": "i2",
        "int16": "i2",
        "int16_t": "i2",
        "uint": "u4",
        "unsigned int": "u4",
        "uint32": "u4",
        "uint32_t": "u4",
        "int": "i4",
        "int32": "i4",
        "int32_t": "i4",
    }
    kind = types.get(header.get("type", "").lower())
    if (kind is None):
        raise ValueError("MRB binary labelmap must contain integer labels.")
    if any((key in header) for key in ("data file", "datafile")):
        raise ValueError("Detached NRRD payloads are not supported; export an embedded seg.nrrd.")
    if ((int(header.get("byte skip", "0")) != 0) or (int(header.get("line skip", "0")) != 0)):
        raise ValueError("NRRD payload byte/line offsets are not supported.")
    endian = header.get("endian", "little").lower()
    if (endian not in ("little", "big")):
        raise ValueError("Invalid NRRD endian.")
    dtype = np.dtype((("<" if (endian == "little") else ">") + kind))
    sizes = _sizes(header)
    expected = (math.prod(sizes) * dtype.itemsize)
    if (expected > (4 * (1024 ** 3))):
        raise ValueError("MRB labelmap exceeds the 4 GiB input safety limit.")
    encoding = header.get("encoding", "raw").lower()
    if (encoding in ("gzip", "gz")):
        with gzip.GzipFile(fileobj = stream) as decoded:
            raw = decoded.read((expected + 1))
    elif (encoding == "raw"):
        raw = stream.read((expected + 1))
    else:
        raise ValueError(f"Unsupported NRRD encoding: {encoding}; use raw or gzip.")
    if (len(raw) != expected):
        raise ValueError(
            f"Truncated or invalid NRRD payload: expected {expected}, got {len(raw)} bytes."
        )
    return np.frombuffer(raw, dtype = dtype).reshape(sizes, order = "F")

def _reference(node, role):
    for item in node.get("references", "").split(";"):
        key, sep, value = item.partition(":")
        if (sep and (key == role)):
            values = value.split()
            if (len(values) != 1):
                raise ValueError(f"Ambiguous MRML {role} reference.")
            return values[0]
    return node.get(("storageNodeRef" if (role == "storage") else role), "")

def _member(scene, filename, names):
    relative = unquote(filename).replace("\\", "/")
    if (
        not relative
        or relative.startswith("/")
        or (":" in relative)
        or (".." in relative.split("/"))
    ):
        raise ValueError("Unsafe MRB storage path.")
    member = posixpath.normpath(posixpath.join(posixpath.dirname(scene), relative))
    if (member not in names):
        raise ValueError(f"MRB storage member is missing: {relative}")
    return member

def _geometry(header):
    sizes = _sizes(header)
    tokens = re.findall(r"none|\([^)]*\)", header.get("space directions", ""), re.I)
    if (len(tokens) != len(sizes)):
        raise ValueError("Missing or invalid NRRD space directions.")
    spatial = [i for i, token in enumerate(tokens) if (token.lower() != "none")]
    if (len(spatial) != 3):
        raise ValueError("MRB segmentation requires exactly three spatial axes.")
    kinds = header.get("kinds", "").split()
    nonspatial = [i for i in range(len(sizes)) if (i not in spatial)]
    if (nonspatial and (
        (len(nonspatial) != 1) or (len(kinds) != len(sizes)) or (kinds[nonspatial[0]] != "list")
    )):
        raise ValueError("Only a single NRRD list/layer axis is supported.")
    vectors = np.column_stack([np.fromstring(tokens[i].strip("()"), sep = ",") for i in spatial])
    origin = np.fromstring(header.get("space origin", "").strip("()"), sep = ",")
    if (
        (vectors.shape != (3, 3))
        or (origin.shape != (3,))
        or not np.all(np.isfinite(vectors))
        or not np.all(np.isfinite(origin))
    ):
        raise ValueError("Invalid NRRD physical geometry.")
    space = header.get("space", "").lower()
    if (space in ("right-anterior-superior", "ras")):
        vectors = (np.diag([-1.0, -1.0, 1.0]) @ vectors)
        origin = (np.diag([-1.0, -1.0, 1.0]) @ origin)
    elif (space not in ("left-posterior-superior", "lps")):
        raise ValueError("Only LPS/RAS MRB coordinate systems are supported.")
    if (abs(np.linalg.det(vectors)) < 1e-12):
        raise ValueError("Singular NRRD geometry.")
    return tuple(sizes[i] for i in spatial), vectors, origin, nonspatial

def _name_for_pipeline(name):
    aliases = {"右冠脉": "RCA", "右冠状动脉": "RCA", "RAM": "RAMUS", "D对角支": "D"}
    return aliases.get(name.strip(), name.strip())

def load_mrb_segmentation(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(set(names)) != len(names)):
            raise ValueError("Duplicate MRB archive member names.")
        scenes = [n for n in names if n.lower().endswith(".mrml")]
        if (len(scenes) != 1):
            raise ValueError("MRB must contain exactly one MRML scene.")
        scene = scenes[0]
        if (archive.getinfo(scene).file_size > (16 * (1024 ** 2))):
            raise ValueError("MRML scene is too large.")
        xml = archive.read(scene)
        if ((b"<!DOCTYPE" in xml.upper()) or (b"<!ENTITY" in xml.upper())):
            raise ValueError("MRML XML entities are not supported.")
        root = ET.fromstring(xml)
        nodes = {e.get("id"): e for e in root.iter() if e.get("id")}
        segmentations = [e for e in root.iter() if (e.tag == "Segmentation")]
        if (len(segmentations) != 1):
            raise ValueError(
                "MRB must contain exactly one segmentation; export the intended labelmap separately."
            )
        segmentation = segmentations[0]
        reference_id = _reference(segmentation, "referenceImageGeometryRef")
        reference = nodes.get(reference_id)
        if (reference is None):
            raise ValueError("MRB segmentation has no referenced source volume geometry.")
        for node in (segmentation, reference):
            if (node.get("transformNodeRef") or _reference(node, "transform")):
                raise ValueError(
                    "MRB parent transform is not supported; harden it in Slicer and re-save."
                )
        storage = nodes.get(_reference(segmentation, "storage"))
        reference_storage = nodes.get(_reference(reference, "storage"))
        if ((storage is None) or (reference_storage is None)):
            raise ValueError("Missing MRB segmentation/reference storage node.")
        member = _member(scene, storage.get("fileName", ""), names)
        reference_member = _member(scene, reference_storage.get("fileName", ""), names)
        with archive.open(reference_member) as stream:
            reference_header = _header(stream)
        with archive.open(member) as stream:
            header = _header(stream)
            array = _payload(stream, header)

    source_shape, directions, origin, layer_axes = _geometry(header)
    target_shape, target_directions, target_origin, target_layers = _geometry(reference_header)
    if (target_layers or not np.allclose(directions, target_directions, rtol = 1e-5, atol = 1e-5)):
        raise ValueError(
            "MRB segmentation/reference geometry differs; export a labelmap in the reference volume geometry."
        )
    offset_float = np.linalg.solve(target_directions, (origin - target_origin))
    offset = np.rint(offset_float).astype(int)
    if (
        not np.allclose(offset_float, offset, atol = 1e-3)
        or np.any((offset < 0))
        or any(((int(offset[i]) + source_shape[i]) > target_shape[i]) for i in range(3))
    ):
        raise ValueError(
            "MRB segmentation crop is outside or not integer-aligned to its reference geometry."
        )
    indices = sorted(
        {int(m.group(1)) for k in header if (m := re.fullmatch(r"Segment(\d+)_Name", k))}
    )
    if (not indices or (len(indices) > 65535)):
        raise ValueError("MRB segmentation is missing usable segment metadata.")
    dtype = (np.uint8 if (len(indices) <= 255) else np.uint16)
    if ((math.prod(target_shape) * np.dtype(dtype).itemsize) > (2 * (1024 ** 3))):
        raise ValueError("MRB reference labelmap exceeds the 2 GiB input safety limit.")
    labels = np.zeros(target_shape, dtype = dtype)
    view = labels[
        tuple(slice(int(offset[i]), (int(offset[i]) + source_shape[i])) for i in range(3))
    ]
    layers = (np.moveaxis(array, layer_axes[0], 0) if layer_axes else array[None, ...])
    entries, segments, seen = {}, [], set()
    overlap_mask = (np.zeros(source_shape, dtype = bool) if (len(layers) > 1) else None)
    for export_value, index in enumerate(indices, 1):
        prefix = f"Segment{index}_"
        name = header[(prefix + "Name")]
        if not name.strip():
            raise ValueError("MRB segment name is empty.")
        layer = int(header.get((prefix + "Layer"), "0"))
        value = int(header[(prefix + "LabelValue")])
        if (not (0 <= layer < len(layers)) or (value <= 0) or ((layer, value) in seen)):
            raise ValueError("Invalid or duplicate MRB (layer, label value) mapping.")
        seen.add((layer, value))
        color = np.fromstring(header.get((prefix + "Color"), ""), sep = " ")
        if (
            (color.shape != (3,))
            or not np.all(np.isfinite(color))
            or np.any(((color < 0) | (color > 1)))
        ):
            raise ValueError("MRB segment is missing a valid RGB color.")
        rgba = (tuple(int(round((v * 255))) for v in color) + (255,))
        mask = (layers[layer] == value)
        if (overlap_mask is not None):
            overlap_mask |= (mask & (view != 0))
        source_count = int(np.count_nonzero(mask))
        view[mask] = export_value
        entries[export_value] = {
            "value": export_value,
            "name": _name_for_pipeline(name),
            "original_name": name,
            "rgba": rgba,
        }
        segments.append(
            {
                "source_index": index,
                "source_name": name,
                "pipeline_name": entries[export_value]["name"],
                "source_layer": layer,
                "source_label_value": value,
                "export_label_value": export_value,
                "source_voxel_count": source_count,
            }
        )
    for layer_index, layer in enumerate(layers):
        known = {value for layer_id, value in seen if (layer_id == layer_index)}
        if not set(int(v) for v in np.unique(layer) if (v != 0)).issubset(known):
            raise ValueError("MRB labelmap contains values without segment metadata.")
    counts = np.bincount(view.ravel(), minlength = (len(indices) + 1))
    for segment in segments:
        segment["export_voxel_count"] = int(counts[segment["export_label_value"]])
    spacing = tuple(float(v) for v in np.linalg.norm(target_directions, axis = 0))
    provenance = {
        "input_compatibility_version": INPUT_COMPATIBILITY_VERSION,
        "label_mapping_source": "mrb_segment_metadata",
        "source_mrb": str(path),
        "mrb_segmentation_member": member,
        "mrb_reference_shape": list(target_shape),
        "mrb_crop_offset": offset.tolist(),
        "mrb_overlap_policy": "segment_index_order_later_overwrites_earlier_as_slicer_labelmap_export",
        "mrb_overlap_voxel_count": (
            int(np.count_nonzero(overlap_mask)) if (overlap_mask is not None) else 0
        ),
        "mrb_segments": segments,
        "mrb_point_labels_present": any(
            ("POINT" in entry["name"].upper()) for entry in entries.values()
        ),
        "mrb_markups_used_as_roots": False,
    }
    return labels, spacing, entries, provenance
