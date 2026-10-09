#!/usr/bin/env python3
"""gazebo_sign_injector - turn a PNG into a Gazebo traffic sign model.

    python3 gazebo_sign_injector.py --name dur --shape octagon --png dur.png
    python3 gazebo_sign_injector.py --manifest signs.csv --out models

Each sign becomes models/<name>/ with model.config, model.sdf, one COLLADA
mesh and the texture. The geometry is generated here (no Blender, no
templates): a flat face carrying the PNG, a thin back plate, a pole and a
base. The face looks towards -Y, so in a world the sign faces traffic coming
from -Y at yaw 0.

Shapes: circle, square, rectangle, triangle, triangle_down, diamond, octagon.
The PNG is mapped onto the shape's bounding box: for a circle draw the sign
on a square canvas touching all four edges; for a rectangle use the same
aspect ratio as --size W H.

Manifest CSV (lines starting with # are ignored, paths are relative to the
CSV file):  name,shape,png[,width[,height]]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__version__ = "0.1.0"

SHAPES = ("circle", "square", "rectangle", "triangle", "triangle_down", "diamond", "octagon")
_TR = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


class SignError(ValueError):
    pass


def slug(name: str) -> str:
    """'Dur Levhası' -> 'dur_levhasi'."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().translate(_TR)).strip("_").lower()
    if not s or s[0].isdigit():
        raise SignError(f"cannot make a model name from {name!r}")
    return s


# ======================================================================
# Geometry
# ======================================================================

Vec2 = Tuple[float, float]


def outline(shape: str, w: float, h: float) -> List[Vec2]:
    """Counter-clockwise (x, z) outline centred on the origin."""
    if shape == "circle":
        r = w / 2
        return [(r * math.cos(2 * math.pi * i / 64), r * math.sin(2 * math.pi * i / 64)) for i in range(64)]
    if shape in ("square", "rectangle"):
        return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    if shape == "triangle":  # equilateral, point up, centred on its bounding box
        th = w * math.sqrt(3) / 2
        return [(-w / 2, -th / 2), (w / 2, -th / 2), (0.0, th / 2)]
    if shape == "triangle_down":
        th = w * math.sqrt(3) / 2
        return [(-w / 2, th / 2), (0.0, -th / 2), (w / 2, th / 2)]
    if shape == "diamond":
        return [(0.0, -w / 2), (w / 2, 0.0), (0.0, w / 2), (-w / 2, 0.0)]
    if shape == "octagon":  # flat top, width across flats = w
        r = w / (2 * math.cos(math.pi / 8))
        return [(r * math.cos(math.pi / 8 + i * math.pi / 4), r * math.sin(math.pi / 8 + i * math.pi / 4))
                for i in range(8)]
    raise SignError(f"unknown shape {shape!r}; use one of {', '.join(SHAPES)}")


@dataclass
class Mesh:
    positions: List[Tuple[float, float, float]]
    normals: List[Tuple[float, float, float]]
    uvs: List[Vec2]
    tris: List[Tuple[int, int, int]]  # indices into positions/normals/uvs (one per corner)

    def add_tri(self, a, b, c, n, uv=None) -> None:
        base = len(self.positions)
        self.positions += [a, b, c]
        self.normals += [n, n, n]
        self.uvs += list(uv) if uv else [(0.0, 0.0)] * 3
        self.tris.append((base, base + 1, base + 2))

    def add_quad(self, a, b, c, d, n) -> None:
        self.add_tri(a, b, c, n)
        self.add_tri(a, c, d, n)


def _box(m: Mesh, x0, x1, y0, y1, z0, z1) -> None:
    p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    m.add_quad(p[0], p[3], p[2], p[1], (0, 0, -1))
    m.add_quad(p[4], p[5], p[6], p[7], (0, 0, 1))
    m.add_quad(p[0], p[1], p[5], p[4], (0, -1, 0))
    m.add_quad(p[2], p[3], p[7], p[6], (0, 1, 0))
    m.add_quad(p[1], p[2], p[6], p[5], (1, 0, 0))
    m.add_quad(p[3], p[0], p[4], p[7], (-1, 0, 0))


@dataclass
class SignSpec:
    name: str
    shape: str
    png: str
    width: float = 0.6
    height: Optional[float] = None  # rectangle only; others derive it from the shape
    center_z: float = 1.9           # height of the sign centre above the ground
    pole: bool = True
    base: bool = True
    thickness: float = 0.02

    def size(self) -> Tuple[float, float]:
        if self.shape == "rectangle":
            return self.width, (self.height or self.width)
        pts = outline(self.shape, self.width, self.width)
        return self.width, max(p[1] for p in pts) - min(p[1] for p in pts)

    def validate(self) -> None:
        slug(self.name)
        if self.shape not in SHAPES:
            raise SignError(f"{self.name}: unknown shape {self.shape!r}; use one of {', '.join(SHAPES)}")
        if not os.path.isfile(self.png):
            raise SignError(f"{self.name}: PNG not found: {self.png}")
        with open(self.png, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                raise SignError(f"{self.name}: {self.png} is not a PNG file")
        w, h = self.size()
        if not 0.05 <= w <= 5 or not 0.05 <= h <= 5:
            raise SignError(f"{self.name}: size must be between 0.05 and 5 m")
        if self.pole and self.center_z - h / 2 < 0.1:
            raise SignError(f"{self.name}: --height {self.center_z} puts the sign into the ground")


def build_meshes(spec: SignSpec) -> Tuple[Mesh, Mesh, Mesh]:
    """(face, back plate with rim, pole and base). Face normal is -Y."""
    w, h = spec.size()
    pts = outline(spec.shape, w, h)
    cz, t = spec.center_z, spec.thickness
    yf, yb = -t, 0.0  # front at y=-t, back at y=0
    face, back, stand = Mesh([], [], [], []), Mesh([], [], [], []), Mesh([], [], [], [])

    # Face: fan around the centre. UV = position inside the bounding box, so
    # the PNG's right edge is at +X and its top edge at +Z (seen from -Y).
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    bx0, bx1, bz0, bz1 = min(xs), max(xs), min(zs), max(zs)

    def uv(x: float, z: float) -> Vec2:
        return ((x - bx0) / (bx1 - bx0), (z - bz0) / (bz1 - bz0))

    c = (sum(xs) / len(xs), sum(zs) / len(zs)) if spec.shape == "circle" else ((bx0 + bx1) / 2, (bz0 + bz1) / 2)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        face.add_tri((c[0], yf, cz + c[1]), (a[0], yf, cz + a[1]), (b[0], yf, cz + b[1]), (0, -1, 0),
                     [uv(*c), uv(*a), uv(*b)])
        # back plate (reverse winding) and rim
        back.add_tri((c[0], yb, cz + c[1]), (b[0], yb, cz + b[1]), (a[0], yb, cz + a[1]), (0, 1, 0))
        ex, ez = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(ex, ez)
        n = (ez / ln, 0.0, -ex / ln)  # outward for a counter-clockwise outline
        back.add_quad((a[0], yf, cz + a[1]), (a[0], yb, cz + a[1]), (b[0], yb, cz + b[1]), (b[0], yf, cz + b[1]), n)

    if spec.pole:
        top = cz + min(h / 2 - 0.05, 0.25)
        _box(stand, -0.02, 0.02, yb, yb + 0.04, 0.0, top)
    if spec.base:
        _box(stand, -0.2, 0.2, yb + 0.02 - 0.2, yb + 0.02 + 0.2, 0.0, 0.02)
    return face, back, stand


# ======================================================================
# Writers
# ======================================================================

def _floats(vals) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def _geometry(gid: str, mesh: Mesh, material: str, with_uv: bool) -> str:
    n = len(mesh.positions)
    sources = f"""      <source id="{gid}-pos">
        <float_array id="{gid}-pos-array" count="{3 * n}">{_floats(c for p in mesh.positions for c in p)}</float_array>
        <technique_common><accessor source="#{gid}-pos-array" count="{n}" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <source id="{gid}-nrm">
        <float_array id="{gid}-nrm-array" count="{3 * n}">{_floats(c for p in mesh.normals for c in p)}</float_array>
        <technique_common><accessor source="#{gid}-nrm-array" count="{n}" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>"""
    inputs = f"""        <input semantic="VERTEX" source="#{gid}-vtx" offset="0"/>
        <input semantic="NORMAL" source="#{gid}-nrm" offset="1"/>"""
    if with_uv:
        sources += f"""
      <source id="{gid}-uv">
        <float_array id="{gid}-uv-array" count="{2 * n}">{_floats(c for p in mesh.uvs for c in p)}</float_array>
        <technique_common><accessor source="#{gid}-uv-array" count="{n}" stride="2">
          <param name="S" type="float"/><param name="T" type="float"/>
        </accessor></technique_common>
      </source>"""
        inputs += f'\n        <input semantic="TEXCOORD" source="#{gid}-uv" offset="2" set="0"/>'
    # Separate offsets per input (the layout Blender writes), each repeating the corner index.
    per_corner = 3 if with_uv else 2
    idx = " ".join(" ".join([str(i)] * per_corner) for t in mesh.tris for i in t)
    return f"""  <geometry id="{gid}" name="{gid}">
    <mesh>
{sources}
      <vertices id="{gid}-vtx"><input semantic="POSITION" source="#{gid}-pos"/></vertices>
      <triangles material="{material}" count="{len(mesh.tris)}">
{inputs}
        <p>{idx}</p>
      </triangles>
    </mesh>
  </geometry>"""


def _effect(eid: str, diffuse: str, ambient: str) -> str:
    return f"""  <effect id="{eid}"><profile_COMMON><technique sid="common"><lambert>
    <ambient><color>{ambient}</color></ambient>
    <diffuse>{diffuse}</diffuse>
  </lambert></technique></profile_COMMON></effect>"""


def write_dae(path: str, spec: SignSpec, token: str) -> None:
    face, back, stand = build_meshes(spec)
    tex = f"../materials/textures/{token}.png"
    parts = [("face", face, True), ("back", back, False)] + ([("stand", stand, False)] if stand.tris else [])
    face_effect = f"""  <effect id="{token}-face-effect"><profile_COMMON>
    <newparam sid="{token}-surface"><surface type="2D"><init_from>{token}-image</init_from></surface></newparam>
    <newparam sid="{token}-sampler"><sampler2D><source>{token}-surface</source></sampler2D></newparam>
    <technique sid="common"><lambert>
      <ambient><color>1 1 1 1</color></ambient>
      <diffuse><texture texture="{token}-sampler" texcoord="UVMap"/></diffuse>
    </lambert></technique>
  </profile_COMMON></effect>"""
    bind = "\n".join(
        f"""      <instance_geometry url="#{token}-{name}"><bind_material><technique_common>
        <instance_material symbol="{token}-{name}-material" target="#{token}-{name}-material">"""
        + ('\n          <bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/>' if uv else "")
        + "\n        </instance_material>\n      </technique_common></bind_material></instance_geometry>"
        for name, _, uv in parts)
    materials = "\n".join(
        f'  <material id="{token}-{n}-material" name="{token}-{n}-material">'
        f'<instance_effect url="#{token}-{n}-effect"/></material>' for n in ("face", "back", "stand"))
    dae = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset>
  <contributor><authoring_tool>gazebo_sign_injector {__version__}</authoring_tool></contributor>
  <unit name="meter" meter="1"/>
  <up_axis>Z_UP</up_axis>
</asset>
<library_images>
  <image id="{token}-image" name="{token}-image"><init_from>{tex}</init_from></image>
</library_images>
<library_effects>
{face_effect}
{_effect(f"{token}-back-effect", '<color>0.25 0.25 0.27 1</color>', '0.25 0.25 0.27 1')}
{_effect(f"{token}-stand-effect", '<color>0.55 0.55 0.55 1</color>', '0.55 0.55 0.55 1')}
</library_effects>
<library_materials>
{materials}
</library_materials>
<library_geometries>
{chr(10).join(_geometry(f"{token}-{n}", m, f"{token}-{n}-material", uv) for n, m, uv in parts)}
</library_geometries>
<library_visual_scenes>
  <visual_scene id="scene" name="scene">
    <node id="{token}" name="{token}">
{bind}
    </node>
  </visual_scene>
</library_visual_scenes>
<scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(dae)


def write_sdf(path: str, spec: SignSpec, token: str) -> None:
    w, h = spec.size()
    t = spec.thickness
    cols = [f"""      <collision name="plate">
        <pose>0 {-t / 2:.4f} {spec.center_z:.4f} 0 0 0</pose>
        <geometry><box><size>{w:.4f} {t:.4f} {h:.4f}</size></box></geometry>
      </collision>"""]
    if spec.pole:
        top = spec.center_z + min(h / 2 - 0.05, 0.25)
        cols.append(f"""      <collision name="pole">
        <pose>0 0.02 {top / 2:.4f} 0 0 0</pose>
        <geometry><box><size>0.04 0.04 {top:.4f}</size></box></geometry>
      </collision>""")
    if spec.base:
        cols.append("""      <collision name="base">
        <pose>0 0.02 0.01 0 0 0</pose>
        <geometry><box><size>0.4 0.4 0.02</size></box></geometry>
      </collision>""")
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{token}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><mesh><uri>model://{token}/meshes/{token}.dae</uri></mesh></geometry>
      </visual>
{chr(10).join(cols)}
    </link>
  </model>
</sdf>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(sdf)


def write_config(path: str, spec: SignSpec, token: str) -> None:
    w, h = spec.size()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{token}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <description>Traffic sign {token}: {spec.shape}, {w:g} x {h:g} m, generated by gazebo_sign_injector</description>
</model>
""")


def make_sign(spec: SignSpec, out_root: str, force: bool = False) -> str:
    spec.validate()
    token = slug(spec.name)
    model_dir = os.path.join(out_root, token)
    if os.path.exists(model_dir):
        if not force:
            raise SignError(f"{model_dir} already exists (use --force to replace it)")
        shutil.rmtree(model_dir)
    os.makedirs(os.path.join(model_dir, "meshes"))
    os.makedirs(os.path.join(model_dir, "materials", "textures"))
    # The texture file name must be unique across models: Gazebo Classic caches
    # textures by file name, so two signs with "sign.png" would look the same.
    shutil.copyfile(spec.png, os.path.join(model_dir, "materials", "textures", f"{token}.png"))
    write_dae(os.path.join(model_dir, "meshes", f"{token}.dae"), spec, token)
    write_sdf(os.path.join(model_dir, "model.sdf"), spec, token)
    write_config(os.path.join(model_dir, "model.config"), spec, token)
    return model_dir


# ======================================================================
# Command line
# ======================================================================

def read_manifest(path: str, defaults: SignSpec) -> List[SignSpec]:
    base = os.path.dirname(os.path.abspath(path))
    specs: List[SignSpec] = []
    with open(path, newline="", encoding="utf-8") as f:
        for lineno, row in enumerate(csv.reader(f), 1):
            row = [c.strip() for c in row]
            if not row or not row[0] or row[0].startswith("#"):
                continue
            if len(row) < 3:
                raise SignError(f"{path}:{lineno}: expected name,shape,png[,width[,height]]")
            try:
                width = float(row[3]) if len(row) > 3 and row[3] else defaults.width
                height = float(row[4]) if len(row) > 4 and row[4] else defaults.height
            except ValueError as exc:
                raise SignError(f"{path}:{lineno}: width/height must be numbers") from exc
            png = row[2] if os.path.isabs(row[2]) else os.path.join(base, row[2])
            specs.append(SignSpec(row[0], row[1], png, width, height, defaults.center_z,
                                  defaults.pole, defaults.base))
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="gazebo_sign_injector.py",
                                description="Turn PNG images into Gazebo traffic sign models.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--manifest", help="CSV with name,shape,png[,width[,height]] per line")
    p.add_argument("--name", help="model name (single sign)")
    p.add_argument("--shape", choices=SHAPES, help="sign shape (single sign)")
    p.add_argument("--png", help="face image (single sign)")
    p.add_argument("--size", nargs="+", type=float, metavar="M",
                   help="width in metres (default 0.6); rectangle: WIDTH HEIGHT")
    p.add_argument("--height", type=float, default=1.9, help="height of the sign centre (default 1.9 m)")
    p.add_argument("--no-pole", action="store_true", help="plate only, no pole")
    p.add_argument("--no-base", action="store_true", help="no base plate at the foot of the pole")
    p.add_argument("--out", default="models", help="output directory (default ./models)")
    p.add_argument("--force", action="store_true", help="replace existing model folders")
    a = p.parse_args(argv)

    width, height = 0.6, None
    if a.size:
        if len(a.size) > 2:
            p.error("--size takes WIDTH or WIDTH HEIGHT")
        width = a.size[0]
        height = a.size[1] if len(a.size) == 2 else None
    defaults = SignSpec("x", "square", "", width, height, a.height, not a.no_pole, not (a.no_base or a.no_pole))

    try:
        if a.manifest:
            if a.name or a.shape or a.png:
                p.error("use either --manifest or --name/--shape/--png")
            specs = read_manifest(a.manifest, defaults)
        else:
            if not (a.name and a.shape and a.png):
                p.error("give --name, --shape and --png (or --manifest)")
            specs = [SignSpec(a.name, a.shape, a.png, width, height, a.height, defaults.pole, defaults.base)]
        # Check everything before writing anything.
        tokens = {}
        for s in specs:
            s.validate()
            tok = slug(s.name)
            if tok in tokens:
                raise SignError(f"{s.name!r} and {tokens[tok]!r} both become model name {tok!r}")
            tokens[tok] = s.name
            if os.path.exists(os.path.join(a.out, tok)) and not a.force:
                raise SignError(f"{os.path.join(a.out, tok)} already exists (use --force to replace it)")
        os.makedirs(a.out, exist_ok=True)
        for s in specs:
            d = make_sign(s, a.out, force=a.force)
            w, h = s.size()
            print(f"{d}  ({s.shape}, {w:g} x {h:g} m)")
    except (SignError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
