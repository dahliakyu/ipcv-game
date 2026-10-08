"""Convert the rigged avatar FBX to a binary glTF (.glb) that Panda3D loads.

Run with Blender (not the conda env), from the repo root:

    blender -b -P tools/convert_avatar.py

The FBX references its textures by an absolute Windows path, so after import
we re-link the colour maps from resource/.../textures by material name. Only
colour: glTF needs roughness and metallic packed into one image, and for this
stylised character a constant matte value looks the same. The .glb embeds
the textures and keeps the armature and skin weights, so the game can pose
the bones in code.
"""

from pathlib import Path

import bpy

SRC = Path("resource/kongzoomaid-cat-seacret")
FBX = SRC / "source" / "MaidCat(Normal).fbx"
OUT = Path("resource/avatar.glb")

# material name -> texture file prefix (the asset spells "Clothers" for colour)
COLOR_MAPS = {
    "Body": "Body_Color(Seacret)",
    "Clothes": "Clothers_Color(Seacret)",
    "Hair": "Hair_Color(Seacret)",
}


def link_color(mat: bpy.types.Material, stem: str) -> None:
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for n in [n for n in nodes if n.type in ("TEX_IMAGE", "NORMAL_MAP")]:
        nodes.remove(n)  # stale node pointing at the missing Windows path
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str((SRC / "textures" / f"{stem}.png").resolve()))
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.7


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(FBX))
    for name, stem in COLOR_MAPS.items():
        link_color(bpy.data.materials[name], stem)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT.resolve()),
        export_format="GLB",
        export_skins=True,
        export_animations=False,
        export_yup=True,
    )
    print(f"wrote {OUT}")


main()
