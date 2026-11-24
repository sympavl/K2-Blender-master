"""
K2 engine model (.model) and animation clip (.clip) exporter for Blender.
Writes SMDL v3 meshes and CLIP v2 animations.
Ported from the original S2 Games 3ds Max exporter (s2exporter.cpp).

Tested with Blender 4.x and 5.x
"""

import bpy
import bmesh
import shutil
from io import BytesIO
import struct
import os
from math import degrees
from mathutils import Matrix

try:
    from .k2_common import (
        log, vlog, dlog, err,
        bone_depth,
        MKEY_X, MKEY_Y, MKEY_Z,
        MKEY_PITCH, MKEY_ROLL, MKEY_YAW,
        MKEY_VISIBILITY,
        MKEY_SCALE_X, MKEY_SCALE_Y, MKEY_SCALE_Z,
        MKEY_COUNT,
    )
except ImportError:
    from k2_common import (
        log, vlog, dlog, err,
        bone_depth,
        MKEY_X, MKEY_Y, MKEY_Z,
        MKEY_PITCH, MKEY_ROLL, MKEY_YAW,
        MKEY_VISIBILITY,
        MKEY_SCALE_X, MKEY_SCALE_Y, MKEY_SCALE_Z,
        MKEY_COUNT,
    )

# Blender version compatibility helpers
try:
    from .k2_compat import (
        bmesh_faces, bmesh_ensure_lookup, bmesh_color_layer, bmesh_deform_layer,
        read_deform_weights,
        mesh_polygons, mesh_vertices, mesh_uv_layers, mesh_color_layers,
        get_uv_layer_values, set_uv_layer_value,
        material_uses_nodes,
        BLENDER_VERSION, BLENDER_4_PLUS, BLENDER_5_PLUS,
    )
except ImportError:
    from k2_compat import (
        bmesh_faces, bmesh_ensure_lookup, bmesh_color_layer, bmesh_deform_layer,
        read_deform_weights,
        mesh_polygons, mesh_vertices, mesh_uv_layers, mesh_color_layers,
        get_uv_layer_values, set_uv_layer_value,
        material_uses_nodes,
        BLENDER_VERSION, BLENDER_4_PLUS, BLENDER_5_PLUS,
    )

# ============================================================================
# Blender version detection
# ============================================================================
# BLENDER_VERSION / BLENDER_4_PLUS / BLENDER_5_PLUS come from k2_compat.

def _get_bmesh_verts(bm):
    """Get vertices from bmesh - compatible with Blender 4.x+"""
    return bm.verts if hasattr(bm, 'verts') else bm.verts

def _get_bmesh_faces(bm):
    """
    Get faces from bmesh.

    Blender 4.x briefly exposed ``bm.polygons`` as an alias, but it was
    removed again in Blender 5.x — ``bm.faces`` is the only portable name.
    """
    return bmesh_faces(bm)

def _get_mesh_vertices(mesh):
    """Get mesh vertices - Blender 4.0+ uses 'vertices' instead of 'verts'"""
    return mesh.vertices if hasattr(mesh, 'vertices') else mesh.verts

def _get_mesh_polygons(mesh):
    """Get mesh polygons - Blender 4.0+ uses 'polygons' instead of 'faces'"""
    return mesh.polygons if hasattr(mesh, 'polygons') else mesh.faces

# ============================================================================
# Export settings container (mirrors K2ExportSettings PropertyGroup)
# ============================================================================

class ExportOptions:
    """Plain data class to pass export settings from operators to export functions."""
    __slots__ = (
        'apply_modifiers', 'force_static', 'remove_hierarchy',
        'copy_textures', 'export_geometry', 'export_model_def', 'export_materials',
        'export_animation', 'frame_start', 'frame_end', 'export_mode',
    )

    def __init__(self, **kw):
        self.apply_modifiers = kw.get('apply_modifiers', True)
        self.force_static = kw.get('force_static', False)
        self.remove_hierarchy = kw.get('remove_hierarchy', False)
        self.copy_textures = kw.get('copy_textures', True)
        self.export_geometry = kw.get('export_geometry', True)
        self.export_model_def = kw.get('export_model_def', True)
        self.export_materials = kw.get('export_materials', True)
        self.export_animation = kw.get('export_animation', False)
        self.frame_start = kw.get('frame_start', 0)
        self.frame_end = kw.get('frame_end', 250)
        self.export_mode = kw.get('export_mode', 'STATIC')

    @staticmethod
    def from_scene(scene):
        s = scene.k2_export_settings
        # export_mode controls force_static: STATIC = True, ANIMATED = False
        force_static = s.export_mode == 'STATIC'
        return ExportOptions(
            apply_modifiers=s.apply_modifiers,
            force_static=force_static,
            remove_hierarchy=s.remove_hierarchy,
            copy_textures=s.copy_textures,
            export_geometry=s.export_geometry,
            export_model_def=s.export_model_def,
            export_materials=s.export_materials,
            export_animation=s.export_animation,
            frame_start=s.frame_start,
            frame_end=s.frame_end,
            export_mode=s.export_mode,
        )


# ============================================================================
# Binary writing helpers
# ============================================================================

def write_block(file, name, data):
    file.write(name.encode('utf8')[:4])
    file.write(struct.pack("<i", len(data)))
    file.write(data)


# ============================================================================
# Bounding box
# ============================================================================

def generate_bbox(mesh_objs):
    """Generate bounding box from mesh objects (supports Blender 4.x+ API)"""
    xx, yy, zz = [], [], []
    for mesh_obj in mesh_objs:
        # mesh_obj can be either a bmesh or a Mesh data block
        verts = None
        if hasattr(mesh_obj, 'verts'):
            verts = mesh_obj.verts
        elif hasattr(mesh_obj, 'vertices'):
            verts = mesh_obj.vertices
        elif hasattr(mesh_obj, '__iter__'):
            # It's a list of vertices
            verts = mesh_obj
            
        if verts:
            for v in verts:
                co = v.co if hasattr(v, 'co') else v
                xx.append(co[0])
                yy.append(co[1])
                zz.append(co[2])
    if not xx:
        return [0, 0, 0, 0, 0, 0]
    return [min(xx), min(yy), min(zz), max(xx), max(yy), max(zz)]


# ============================================================================
# Chunk data builders
# ============================================================================

def create_mesh_data(mesh, vert, index, name, mname, bone_link=-1, mode=1):
    buf = BytesIO()
    buf.write(struct.pack("<i", index))
    buf.write(struct.pack("<i", mode))  # mode: 1 = skinned blended, 2 = skinned non-blended
    buf.write(struct.pack("<i", len(vert)))
    buf.write(struct.pack("<6f", *generate_bbox([mesh])))
    buf.write(struct.pack("<i", bone_link))
    buf.write(struct.pack("<B", len(name)))
    buf.write(struct.pack("<B", len(mname)))
    buf.write(name)
    buf.write(struct.pack("<B", 0))
    buf.write(mname)
    buf.write(struct.pack("<B", 0))
    return buf.getvalue()


def create_vrts_data(verts, meshindex):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    for v in verts:
        buf.write(struct.pack("<3f", *v.co))
    return buf.getvalue()


def create_face_data(verts, faces, meshindex):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    buf.write(struct.pack("<i", len(faces)))
    if len(verts) < 255:
        buf.write(struct.pack("<B", 1))
        fmt = '<3B'
    else:
        buf.write(struct.pack("<B", 2))
        fmt = '<3H'
    for f in faces:
        if len(f) == 3:
            buf.write(struct.pack(fmt, *f))
        else:
            log(f"Warning: Face {f} is not a triangle and will be skipped")
    return buf.getvalue()


def create_tang_data(tang, meshindex, channel=0):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    buf.write(struct.pack("<i", channel))
    for t in tang:
        buf.write(struct.pack('<3f', *list(t)))
    return buf.getvalue()


def create_texc_data(texc, meshindex, channel=0):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    buf.write(struct.pack("<i", channel))
    for t in texc:
        buf.write(struct.pack("<2f", t[0], 1.0 - t[1]))
    return buf.getvalue()


def create_colr_data(colr, meshindex):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    for c in colr:
        # Colours may be plain RGBA tuples/vectors (Blender 5.x bmesh colour
        # layers) or objects exposing .r/.g/.b/.a.
        if hasattr(c, 'r'):
            r, g, b, a = c.r, c.g, c.b, c.a
        else:
            r, g, b = c[0], c[1], c[2]
            a = c[3] if len(c) > 3 else 1.0
        buf.write(struct.pack("<4B",
                              int(max(0.0, min(1.0, r)) * 255),
                              int(max(0.0, min(1.0, g)) * 255),
                              int(max(0.0, min(1.0, b)) * 255),
                              int(max(0.0, min(1.0, a)) * 255)))
    return buf.getvalue()


def create_nrml_data(verts, meshindex):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    for v in verts:
        buf.write(struct.pack("<3f", *v.normal))
    return buf.getvalue()


def create_lnk1_data(lnk1, meshindex, bone_indices):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    buf.write(struct.pack("<i", len(lnk1)))
    for influences in lnk1:
        influences = [inf for inf in influences if inf[0] in bone_indices]
        count = len(influences)
        buf.write(struct.pack("<i", count))
        if count > 0:
            buf.write(struct.pack(f'<{count}f', *[inf[1] for inf in influences]))
            buf.write(struct.pack(f'<{count}I', *[bone_indices[inf[0]] for inf in influences]))
    return buf.getvalue()


def create_lnk2_data(lnk2, meshindex, bone_index):
    """Create single link data (lnk2) - one bone per vertex."""
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    buf.write(struct.pack("<i", len(lnk2)))
    for bone_idx in lnk2:
        buf.write(struct.pack("<i", bone_idx))
        buf.write(struct.pack("<i", bone_index))
    return buf.getvalue()


def create_sign_data(meshindex, sign):
    buf = BytesIO()
    buf.write(struct.pack("<i", meshindex))
    buf.write(struct.pack("<i", 0))
    for s in sign:
        buf.write(struct.pack("<b", s))
    return buf.getvalue()


# ============================================================================
# Face / vertex data conversion
# ============================================================================

def calc_face_signs(ftexc):
    fsigns = []
    for uv in ftexc:
        cross = ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[1][1])
                 - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[1][0]))
        if cross > 0:
            fsigns.append((0, 0, 0))
        else:
            fsigns.append((-1, -1, -1))
    return fsigns


def face_to_vertices(faces, fdata, verts):
    """Map per-face data to per-vertex data. First-write wins for shared vertices."""
    vdata = [None] * len(verts)
    for fi, f in enumerate(faces):
        if fi >= len(fdata):
            log(f"face_to_vertices: face index {fi} out of range (fdata length {len(fdata)})")
            continue
        face_data = fdata[fi]
        if len(f) != len(face_data):
            log(f"face_to_vertices: mismatch at face {fi} - {len(f)} vs {len(face_data)}")
            continue
        for vi, v in enumerate(f):
            vdata[v] = face_data[vi]
    return vdata


def face_to_vertices_dup(faces, fdata, verts):
    """Map per-face data to per-vertex, duplicating vertices when values conflict."""
    vdata = [None] * len(verts)
    for fi, f in enumerate(faces):
        if fi >= len(fdata):
            log(f"face_to_vertices_dup: face index {fi} out of range (fdata length {len(fdata)})")
            continue
        face_data = fdata[fi]
        if len(f) != len(face_data):
            log(f"face_to_vertices_dup: mismatch at face {fi} - {len(f)} vs {len(face_data)}")
            continue
        for vi, v in enumerate(f):
            if vdata[v] is None or vdata[v] == face_data[vi]:
                vdata[v] = face_data[vi]
            else:
                newind = len(verts)
                verts.append(verts[v])
                faces[fi][vi] = newind
                vdata.append(face_data[vi])
    return vdata


# ============================================================================
# Bone data builder
# ============================================================================

def create_bone_data(armature, arm_matrix, transform, flatten_hierarchy=False):
    """Build bone chunk data. If flatten_hierarchy is True, all bones parent to root."""
    bones = sorted(armature.bones.values(), key=bone_depth)
    bone_names = [bone.name for bone in bones]

    bonedata = BytesIO()
    for bone in bones:
        base = bone.matrix_local.copy()
        if transform:
            base = base @ arm_matrix
        base_inv = base.copy()
        base_inv.invert()

        if flatten_hierarchy:
            parent_index = -1
        else:
            parent_index = bone_names.index(bone.parent.name) if bone.parent else -1

        base_inv.transpose()
        base.transpose()

        bonedata.write(struct.pack("<i", parent_index))
        bonedata.write(struct.pack('<12f', *sum([list(row[0:3]) for row in base_inv], [])))
        bonedata.write(struct.pack('<12f', *sum([list(row[0:3]) for row in base], [])))
        name_bytes = bone.name.encode('utf8')
        bonedata.write(struct.pack("B", len(name_bytes)))
        bonedata.write(name_bytes)
        bonedata.write(struct.pack("B", 0))

    return bone_names, bonedata.getvalue()


# ============================================================================
# Selection helpers
# ============================================================================

def _select_objects_by_type(*types):
    """
    Deselect all, then select objects matching the given type(s).

    Only objects that are actually part of the current view layer are
    considered — selecting objects from other scenes or from collections that
    are not linked to the view layer makes ``bpy.context.selected_objects``
    disagree with the visible scene, which broke the clip export whenever
    more than one armature existed in the blend file.
    """
    view_layer = bpy.context.view_layer
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError:
        pass

    for obj in view_layer.objects:
        obj.select_set(False)

    view_layer.objects.active = None

    found = set()
    for obj in view_layer.objects:
        if obj.type in types:
            obj.select_set(True)
            found.add(obj.type)

    for t in types:
        if t not in found:
            log(f"No {t.lower()} objects found in the scene.")


# ============================================================================
# Texture copy helper
# ============================================================================

def _copy_textures(meshes_objs, export_dir):
    """Copy texture files referenced by materials to the export directory."""
    copied = set()
    textures_dir = os.path.join(export_dir, "textures")
    os.makedirs(textures_dir, exist_ok=True)
    for obj in meshes_objs:
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            if not mat or not material_uses_nodes(mat):
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image and node.image.filepath:
                    src = bpy.path.abspath(node.image.filepath)
                    if src in copied or not os.path.isfile(src):
                        continue
                    dst = os.path.join(textures_dir, os.path.basename(src))
                    try:
                        shutil.copy2(src, dst)
                        vlog(f"Copied texture: {os.path.basename(src)}")
                        copied.add(src)
                    except Exception as e:
                        log(f"Failed to copy texture {src}: {e}")


def _collect_export_materials(mesh_objs):
    """Return unique Blender materials used by exported meshes, preserving first-seen order."""
    materials = []
    seen = set()
    for obj in mesh_objs:
        for mat in obj.data.materials:
            if not mat:
                continue
            key = mat.name_full
            if key in seen:
                continue
            seen.add(key)
            materials.append(mat)
    return materials


def _material_export_name(index):
    return "material" if index == 0 else f"material{index + 1}"


def _node_image_path(node):
    if not node or node.type != 'TEX_IMAGE' or not node.image:
        return None
    image_path = bpy.path.abspath(node.image.filepath_raw or node.image.filepath)
    if not image_path:
        return None
    return image_path.replace("\\", "/")


def _looks_like_tex_type(text, tex_type):
    text = (text or "").lower()
    if tex_type == 'color':
        return any(k in text for k in ('color', 'diff', 'diffuse', 'albedo'))
    if tex_type == 'color2':
        return any(k in text for k in ('color2', 'diff2', 'team', 'tint'))
    if tex_type == 'color3':
        return any(k in text for k in ('color3', 'diff3', 'detail', 'spec'))
    if tex_type == 'normal':
        return any(k in text for k in ('normal', 'norm', 'nrm'))
    if tex_type == 'mrao':
        return any(k in text for k in ('mrao', 'orm'))
    if tex_type == 'emissive':
        return any(k in text for k in ('emissive', 'emis', 'glow', 'self'))
    return False


def _resolve_material_textures(mat):
    """
    Infer texture roles from image texture nodes.
    The importer labels nodes consistently, but we also fall back to filename matching.
    """
    textures = {}
    if not mat or not material_uses_nodes(mat) or not mat.node_tree:
        return textures

    for node in mat.node_tree.nodes:
        if node.type != 'TEX_IMAGE':
            continue
        image_path = _node_image_path(node)
        if not image_path:
            continue
        filename = os.path.basename(image_path).lower()
        label = (node.label or node.name or "").lower()
        search_text = f"{label} {filename}"
        for tex_type in ('color2', 'color3', 'color', 'normal', 'mrao', 'emissive'):
            if tex_type in textures:
                continue
            if _looks_like_tex_type(search_text, tex_type):
                textures[tex_type] = image_path
                break

    return textures


def _to_export_relpath(path, export_dir):
    if not path:
        return None
    path = bpy.path.abspath(path)
    if export_dir:
        try:
            rel = os.path.relpath(path, export_dir)
            return rel.replace("\\", "/")
        except ValueError:
            pass
    return os.path.basename(path).replace("\\", "/")


def _write_material_file(filepath, textures):
    shader = "/shared/shaders/heroes/hero_diffuse_normal_mrao_team_emissive.shader"
    diffuse = textures.get('color')
    normal = textures.get('normal')
    mrao = textures.get('mrao')
    emissive = textures.get('emissive')
    color3 = textures.get('color3')

    glossiness = "64" if color3 else "32"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<material>',
        f'\t<parameters shader="{shader}" vDiffuseColor="1 1 1" fSpecularLevel="3.0" fGlossiness="{glossiness}" fOpacity="1.0" />',
    ]

    if diffuse:
        lines.extend([
            '\t<phase name="shadow" cull="back" blend="alphatest">',
            f'\t\t<sampler name="diffuse" texture="{diffuse}" />',
            '\t</phase>',
            '\t<phase name="color" cull="back" blend="alphatest">',
            f'\t\t<sampler name="diffuse" texture="{diffuse}" />',
        ])
        if normal:
            lines.append(f'\t\t<sampler name="normalmap" texture="{normal}" />')
        if mrao:
            lines.append(f'\t\t<sampler name="mrao" texture="{mrao}" />')
        if emissive:
            lines.append(f'\t\t<sampler name="emissive" texture="{emissive}" />')
        lines.extend([
            '\t\t<samplercube name="cube" texture="/world/sky/deadlock/1.tga" />',
            '\t</phase>',
            '\t<phase name="fade" cull="back" colorwrite="false" alphawrite="false" depthwrite="true" blend="alphatest">',
            f'\t\t<sampler name="diffuse" texture="{diffuse}" />',
            '\t\t<multipass cull="back" blend="translucent">',
            f'\t\t\t<sampler name="diffuse" texture="{diffuse}" />',
        ])
        if normal:
            lines.append(f'\t\t\t<sampler name="normalmap" texture="{normal}" />')
        if mrao:
            lines.append(f'\t\t\t<sampler name="mrao" texture="{mrao}" />')
        if emissive:
            lines.append(f'\t\t\t<sampler name="emissive" texture="{emissive}" />')
        lines.extend([
            '\t\t\t<samplercube name="cube" texture="/world/sky/deadlock/1.tga" />',
            '\t\t</multipass>',
            '\t</phase>',
        ])

    lines.append('</material>')

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write("\n".join(lines) + "\n")


def _write_model_def_file(filepath, model_filename):
    model_name = os.path.basename(bpy.data.filepath) if bpy.data.filepath else os.path.basename(model_filename)
    model_rel = os.path.basename(model_filename).replace("\\", "/")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model name="{model_name}" file="{model_rel}" type="K2" high="{model_rel}" med="{model_rel}" low="{model_rel}" >',
        '',
        '</model>',
    ]
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write("\n".join(lines) + "\n")


def _export_companion_files(mesh_objs, export_dir, model_filename, write_materials=True,
                            write_model_def=True, copy_textures=False):
    """Write original-style companion .material and .mdf files next to the exported model."""
    if not export_dir:
        return

    materials = _collect_export_materials(mesh_objs)
    material_names = {}
    for index, mat in enumerate(materials):
        stem = _material_export_name(index)
        material_names[mat.name_full] = stem
        if not write_materials:
            continue
        textures = {}
        for tex_type, path in _resolve_material_textures(mat).items():
            tex_basename = os.path.basename(path).replace("\\", "/")
            textures[tex_type] = (
                f"textures/{tex_basename}"
                if copy_textures else _to_export_relpath(path, export_dir)
            )
        _write_material_file(os.path.join(export_dir, f"{stem}.material"), textures)

    if write_model_def:
        model_stem = os.path.splitext(os.path.basename(model_filename))[0]
        _write_model_def_file(os.path.join(export_dir, f"{model_stem}.mdf"), model_filename)

    return material_names


# ============================================================================
# Per-mesh K2 settings helpers
# ============================================================================

def _get_mesh_type(obj):
    """Return the K2 mesh type string for an object."""
    if hasattr(obj, 'k2_mesh_settings'):
        return obj.k2_mesh_settings.mesh_type
    if obj.name.startswith('_surf'):
        return 'COLLISION'
    if obj.name.startswith('_bone'):
        return 'REFBONE'
    return 'NORMAL'


def _should_exclude_normals(obj):
    if hasattr(obj, 'k2_mesh_settings'):
        return obj.k2_mesh_settings.exclude_normals
    return False


# ============================================================================
# Mesh export
# ============================================================================

def export_k2_mesh(filename, opts=None):
    """
    Export scene meshes to a K2 .model file.
    opts: ExportOptions instance (or None for legacy call with apply_modifiers bool).

    Supports Blender 4.x and 5.x APIs.
    """
    if opts is None or isinstance(opts, bool):
        apply_mod = opts if isinstance(opts, bool) else True
        opts = ExportOptions(apply_modifiers=apply_mod)

    # Export the selected meshes/armature when the user made a selection,
    # otherwise export everything in the scene.
    view_layer = bpy.context.view_layer
    preselected = [o for o in view_layer.objects
                   if o.select_get() and o.type in ('MESH', 'ARMATURE')]

    if preselected:
        for obj in view_layer.objects:
            obj.select_set(False)
        for obj in preselected:
            obj.select_set(True)
    else:
        _select_objects_by_type('ARMATURE', 'MESH')

    meshes = []
    mesh_objs = []
    sprites = []
    sprite_objs = []
    armature = None
    arm_matrix = None

    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            mesh_type = _get_mesh_type(obj)
            if mesh_type == 'REFBONE':
                vlog(f"Skipping reference bone '{obj.name}'")
                continue
            elif mesh_type in ('SPRITE', 'GROUND'):
                # Handle sprite/ground plane
                matrix = obj.matrix_world
                if opts.apply_modifiers:
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    me = obj.evaluated_get(depsgraph).to_mesh()
                else:
                    me = obj.data
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.triangulate(bm, faces=list(_get_bmesh_faces(bm)))
                bm.transform(matrix)
                sprites.append((obj, bm, mesh_type))
                sprite_objs.append(obj)
            elif mesh_type in ('NORMAL', 'COLLISION'):
                matrix = obj.matrix_world
                if opts.apply_modifiers:
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    me = obj.evaluated_get(depsgraph).to_mesh()
                else:
                    me = obj.data
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.triangulate(bm, faces=list(_get_bmesh_faces(bm)))
                bm.transform(matrix)
                meshes.append((obj, bm, mesh_type))
                mesh_objs.append(obj)
        elif obj.type == 'ARMATURE':
            armature = obj.data
            arm_matrix = obj.matrix_world

    use_armature = armature and not opts.force_static
    bone_names = []
    bonedata = b''
    # The armature must be in POSE position while the bone rest matrices are
    # read. The previous value is restored afterwards - leaving the armature
    # in REST makes the rig stop reacting to animation.
    prev_pose_position = None
    if use_armature:
        prev_pose_position = armature.pose_position
        armature.pose_position = 'POSE'
        bone_names, bonedata = create_bone_data(
            armature, arm_matrix, opts.apply_modifiers,
            flatten_hierarchy=opts.remove_hierarchy,
        )
        if prev_pose_position != 'POSE':
            armature.pose_position = prev_pose_position

    if not meshes:
        log("WARNING: No mesh objects found in scene - exported model will contain only bones")
    else:
        log(f"Exporting {len(meshes)} mesh(es): {[obj.name for obj, _, _ in meshes]}")

    num_normal_meshes = sum(1 for _, _, mt in meshes if mt == 'NORMAL')
    num_surfs = sum(1 for _, _, mt in meshes if mt == 'COLLISION')
    num_sprites = len(sprites)

    # Build header
    headdata = BytesIO()
    headdata.write(struct.pack("<i", 3))  # version
    headdata.write(struct.pack("<i", num_normal_meshes))
    headdata.write(struct.pack("<i", num_sprites))  # sprites
    headdata.write(struct.pack("<i", num_surfs))
    headdata.write(struct.pack("<i", len(armature.bones) if use_armature else 0))
    all_bm = [bm for _, bm, _ in meshes]
    headdata.write(struct.pack("<6f", *generate_bbox(all_bm)))

    export_dir = os.path.dirname(filename)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    material_names = _export_companion_files(
        mesh_objs,
        export_dir,
        filename,
        write_materials=opts.export_materials,
        write_model_def=opts.export_model_def,
        copy_textures=opts.copy_textures,
    ) or {}

    with open(filename, 'wb') as file:
        file.write(b'SMDL')
        write_block(file, 'head', headdata.getvalue())
        if use_armature:
            write_block(file, 'bone', bonedata)

        meshindex = 0
        for obj, mesh, mesh_type in meshes:
            if not opts.export_geometry and mesh_type == 'NORMAL':
                meshindex += 1
                continue

            vert = list(mesh.verts)
            faces = []
            ftexc = []
            ftang = []
            fcolr = []
            flnk1 = []
            exclude_nrml = _should_exclude_normals(obj)

            # The bmesh index tables must be up to date before we can rely on
            # `loop.vert.index` / `vert.index`.
            bmesh_ensure_lookup(mesh)
            mesh.normal_update()

            # UV layers live on the bmesh loop layers. `mesh.loops.layers.uv`
            # is available on BMesh (unlike on a plain Mesh data-block).
            uv_layers = mesh.loops.layers.uv
            uv_layer_list = list(uv_layers) if uv_layers is not None else []
            uv_lay = uv_layers.active if uv_layers is not None else None
            has_uv = uv_lay is not None

            # Vertex colours: byte-colour layers are exposed through
            # `loops.layers.color`, float-colour layers through
            # `loops.layers.float_color`.
            col_lay = bmesh_color_layer(mesh)
            has_color = col_lay is not None

            # Deform (skin) layer for skinning.
            dvert_lay = bmesh_deform_layer(mesh)
            if dvert_lay and not opts.force_static:
                flnk1 = [
                    read_deform_weights(mesh, v, dvert_lay)
                    for v in mesh.verts
                ]

            mesh_faces = _get_bmesh_faces(mesh)
            for f in mesh_faces:
                uv = []
                col = []
                vindex = []
                tang = []
                for loop in f.loops:
                    if has_uv and uv_lay is not None:
                        uv.append(tuple(loop[uv_lay].uv))
                    vindex.append(loop.vert.index)
                    if has_color and col_lay is not None:
                        # Blender 5.x exposes the colour as a plain vector.
                        col.append(tuple(loop[col_lay]))
                    tang.append(loop.calc_tangent())
                if has_uv:
                    ftexc.append(uv)
                ftang.append(tang)
                faces.append(vindex)
                if has_color:
                    fcolr.append(col)

            texc = None
            tang_data = None
            sign = None
            if has_uv:
                fsign = calc_face_signs(ftexc)
                sign = face_to_vertices(faces, fsign, vert)
                texc = face_to_vertices(faces, ftexc, vert)
                tang_data = face_to_vertices(faces, ftang, vert)
                for i in range(len(vert)):
                    if tang_data[i] is not None:
                        tang_data[i] = tang_data[i] - vert[i].normal * tang_data[i].dot(vert[i].normal)
                        tang_data[i].normalize()

            colr = face_to_vertices(faces, fcolr, vert) if has_color else None

            if obj.data.materials:
                mat = obj.data.materials[0]
                mat_stub = material_names.get(mat.name_full)
                mat_name = (mat_stub or mat.name).encode('utf8')
            else:
                mat_name = obj.name.encode('utf8')

            # Determine mesh mode and bone link
            # Check if mesh has blended weights (multiple bones per vertex)
            has_blended_weights = False
            if flnk1:
                for influences in flnk1:
                    if len(influences) > 1:
                        has_blended_weights = True
                        break

            # Get bone_link from mesh settings (single bone link)
            bone_link = -1
            mesh_mode = 2  # Default to non-blended (mode 2)
            if hasattr(obj, 'k2_mesh_settings'):
                # Check if mesh is linked to a specific bone
                for group in obj.vertex_groups:
                    if group.name in bone_names:
                        bone_link = bone_names.index(group.name)
                        break

            # Set mode based on weights
            if has_blended_weights:
                mesh_mode = 1  # MESH_SKINNED_BLENDED

            write_block(file, 'mesh', create_mesh_data(
                mesh, vert, meshindex, obj.name.encode('utf8'), mat_name,
                bone_link=bone_link, mode=mesh_mode))
            write_block(file, 'vrts', create_vrts_data(vert, meshindex))

            # Write link data based on mode
            if use_armature and flnk1:
                bone_indices_map = {}
                for group in obj.vertex_groups:
                    if group.name in bone_names:
                        bone_indices_map[group.index] = bone_names.index(group.name)

                if has_blended_weights:
                    # Mode 1: MESH_SKINNED_BLENDED - use lnk1
                    write_block(file, 'lnk1', create_lnk1_data(flnk1, meshindex, bone_indices_map))
                elif bone_link >= 0:
                    # Mode 2: MESH_SKINNED_NONBLENDED with bonelink - use lnk2
                    lnk2_indices = [bone_link] * len(vert)
                    write_block(file, 'lnk2', create_lnk2_data(lnk2_indices, meshindex, bone_link))

            if faces:
                write_block(file, 'face', create_face_data(vert, faces, meshindex))
                if has_uv and texc is not None:
                    # Write UV channel 0 data (required)
                    write_block(file, "texc", create_texc_data(texc, meshindex, channel=0))
                    for i in range(len(tang_data)):
                        if sign[i] == 0:
                            tang_data[i] = -(tang_data[i].copy())
                    write_block(file, "tang", create_tang_data(tang_data, meshindex, channel=0))
                    write_block(file, "sign", create_sign_data(meshindex, sign))

                    # Write additional UV channels if available (up to 8)
                    if len(uv_layer_list) > 1:
                        for channel_idx in range(1, min(len(uv_layer_list), 8)):
                            uv_lay = uv_layer_list[channel_idx]
                            ftexc_ch = []
                            ftang_ch = []

                            for f in mesh_faces:
                                uv_ch = []
                                tang_ch = []
                                for loop in f.loops:
                                    uv_ch.append(tuple(loop[uv_lay].uv))
                                    tang_ch.append(loop.calc_tangent())
                                ftexc_ch.append(uv_ch)
                                ftang_ch.append(tang_ch)

                                texc_ch = face_to_vertices(faces, ftexc_ch, vert)
                                tang_ch_data = face_to_vertices(faces, ftang_ch, vert)

                                fsign_ch = calc_face_signs(ftexc_ch)
                                sign_ch = face_to_vertices(faces, fsign_ch, vert)

                                write_block(file, "texc", create_texc_data(texc_ch, meshindex, channel=channel_idx))
                                for i in range(len(tang_ch_data)):
                                    if tang_ch_data[i] is not None:
                                        tang_ch_data[i] = tang_ch_data[i] - vert[i].normal * tang_ch_data[i].dot(vert[i].normal)
                                        tang_ch_data[i].normalize()
                                write_block(file, "tang", create_tang_data(tang_ch_data, meshindex, channel=channel_idx))

                if not exclude_nrml:
                    write_block(file, "nrml", create_nrml_data(vert, meshindex))
            if colr is not None:
                write_block(file, "colr", create_colr_data(colr, meshindex))

            vlog(f'Mesh {meshindex} ({obj.name}): {len(vert) - len(list(mesh.verts))} verts duplicated')
            meshindex += 1

    # Export sprites
    if sprites:
        log(f"Exporting {len(sprites)} sprite(s): {[obj.name for obj, _, _ in sprites]}")
        for obj, mesh, sprite_type in sprites:
            vert = list(mesh.verts)
            bmesh_ensure_lookup(mesh)
            mesh_faces = _get_bmesh_faces(mesh)
            # On a BMesh, `face.verts` yields BMVert objects - the vertex
            # indices are needed for the binary face data.
            faces = [[v.index for v in f.verts] for f in mesh_faces]

            # Get bounding box
            bbox = generate_bbox([mesh])

            # Sprite mode: 3 = billboard, 4 = ground plane
            sprite_mode = 3 if sprite_type == 'SPRITE' else 4

            if obj.data.materials:
                mat = obj.data.materials[0]
                mat_name = mat.name.encode('utf8')
            else:
                mat_name = obj.name.encode('utf8')

            # Write sprite mesh block (similar to normal mesh but with different mode)
            sprite_mesh_data = BytesIO()
            sprite_mesh_data.write(struct.pack("<i", 0))  # sprite index starts at 0
            sprite_mesh_data.write(struct.pack("<i", sprite_mode))  # mode: 3=billboard, 4=ground
            sprite_mesh_data.write(struct.pack("<i", len(vert)))
            sprite_mesh_data.write(struct.pack("<6f", *bbox))
            sprite_mesh_data.write(struct.pack("<i", -1))  # bone link
            sprite_mesh_data.write(struct.pack("<B", len(obj.name.encode('utf8'))))
            sprite_mesh_data.write(struct.pack("<B", len(mat_name)))
            sprite_mesh_data.write(obj.name.encode('utf8'))
            sprite_mesh_data.write(struct.pack("<B", 0))
            sprite_mesh_data.write(mat_name)
            sprite_mesh_data.write(struct.pack("<B", 0))

            write_block(file, 'mesh', sprite_mesh_data.getvalue())

            # Write vertices
            write_block(file, 'vrts', create_vrts_data(vert, 0))

            # Write faces
            if faces:
                write_block(file, 'face', create_face_data(vert, faces, 0))

    if opts.copy_textures and export_dir:
        _copy_textures(mesh_objs + sprite_objs, export_dir)

    log(f"Exported {len(meshes)} mesh(es), {len(sprites)} sprite(s) to {filename}")


# ============================================================================
# Clip export
# ============================================================================

def _write_clip_bone(file, bone_name_bytes, motion, index):
    """Write all motion key channels for a single bone."""
    for keytype in range(MKEY_COUNT):
        keydata = BytesIO()
        key = motion[keytype]
        if min(key) == max(key):
            key = [key[0]]
        numkeys = len(key)
        keydata.write(struct.pack("<i", index))
        keydata.write(struct.pack("<i", keytype))
        keydata.write(struct.pack("<i", numkeys))
        keydata.write(struct.pack("B", len(bone_name_bytes)))
        keydata.write(bone_name_bytes)
        keydata.write(struct.pack("B", 0))
        if keytype == MKEY_VISIBILITY:
            keydata.write(struct.pack(f'{numkeys}B', *key))
        else:
            keydata.write(struct.pack(f'<{numkeys}f', *key))
        write_block(file, 'bmtn', keydata.getvalue())


def export_k2_clip(filename, opts=None, frame_start=None, frame_end=None):
    """
    Export animation to a K2 .clip file.
    opts: ExportOptions instance (or bool for legacy apply_modifiers).
    """
    if opts is None or isinstance(opts, bool):
        transform = opts if isinstance(opts, bool) else True
        opts = ExportOptions(apply_modifiers=transform)
    if frame_start is not None:
        opts.frame_start = frame_start
    if frame_end is not None:
        opts.frame_end = frame_end

    # Respect the user's selection: if exactly one armature is selected, use
    # it. Otherwise fall back to the active object, and only as a last resort
    # select every armature in the scene.
    view_layer = bpy.context.view_layer
    selected = [o for o in view_layer.objects
                if o.select_get() and o.type == 'ARMATURE']
    active = view_layer.objects.active

    if len(selected) == 1:
        arm_ob = selected[0]
    elif active is not None and active.type == 'ARMATURE':
        arm_ob = active
    else:
        _select_objects_by_type('ARMATURE')
        selected = [o for o in view_layer.objects
                    if o.select_get() and o.type == 'ARMATURE']
        if len(selected) != 1:
            err('Select the armature to export')
            return
        arm_ob = selected[0]

    armature = arm_ob.data
    motions = {}

    vlog('Baking animation...')

    world_mat = arm_ob.matrix_world if opts.apply_modifiers else Matrix.Identity(4)
    scene = bpy.context.scene
    pose = arm_ob.pose

    for frame in range(opts.frame_start, opts.frame_end + 1):
        scene.frame_set(frame)
        for bone in pose.bones:
            matrix = bone.matrix
            if bone.parent:
                matrix = bone.parent.matrix.inverted() @ matrix
            if opts.apply_modifiers:
                matrix = world_mat @ matrix

            if bone.name not in motions:
                motions[bone.name] = [[] for _ in range(MKEY_COUNT)]

            motion = motions[bone.name]
            translation = matrix.to_translation()
            rotation = matrix.to_euler('YXZ')
            scale = matrix.to_scale()

            motion[MKEY_X].append(translation[0])
            motion[MKEY_Y].append(translation[1])
            motion[MKEY_Z].append(translation[2])
            motion[MKEY_PITCH].append(degrees(rotation[0]))
            motion[MKEY_ROLL].append(degrees(rotation[1]))
            motion[MKEY_YAW].append(degrees(rotation[2]))
            motion[MKEY_SCALE_X].append(scale[0])
            motion[MKEY_SCALE_Y].append(scale[1])
            motion[MKEY_SCALE_Z].append(scale[2])
            motion[MKEY_VISIBILITY].append(255)

    headdata = BytesIO()
    headdata.write(struct.pack("<i", 2))  # version
    headdata.write(struct.pack("<i", len(motions)))
    headdata.write(struct.pack("<i", opts.frame_end - opts.frame_start + 1))

    export_dir = os.path.dirname(filename)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    with open(filename, 'wb') as file:
        file.write(b'CLIP')
        write_block(file, 'head', headdata.getvalue())

        for index, bone_name in enumerate(
            sorted(armature.bones.keys(), key=lambda x: bone_depth(armature.bones[x]))
        ):
            _write_clip_bone(file, bone_name.encode('utf8'), motions[bone_name], index)

    log(f"Exported clip ({opts.frame_end - opts.frame_start + 1} frames) to {filename}")
