"""
Blender version compatibility helpers for the K2 addon.

Centralises every API difference between Blender 3.x / 4.x / 5.x so the rest
of the addon can stay version-agnostic.

Blender 5.0 notes:
  * Actions became "slotted": assigning ``animation_data.action`` is no longer
    enough, an ``action_slot`` must be assigned as well.
  * The legacy ``action.fcurves`` / ``action.groups`` / ``action.id_root``
    properties were removed. Use the slot's channelbag instead.
  * ``bpy.props`` values no longer live in the same IDProperty container as
    user custom properties, so dict-style access is gone.
  * ``material.use_nodes`` is deprecated (materials always have a node tree).
"""

import bpy

BLENDER_VERSION = bpy.app.version
BLENDER_3_PLUS = BLENDER_VERSION >= (3, 0, 0)
BLENDER_4_PLUS = BLENDER_VERSION >= (4, 0, 0)
BLENDER_4_1_PLUS = BLENDER_VERSION >= (4, 1, 0)
BLENDER_4_4_PLUS = BLENDER_VERSION >= (4, 4, 0)
BLENDER_5_PLUS = BLENDER_VERSION >= (5, 0, 0)


# ============================================================================
# Actions / animation slots
# ============================================================================

def create_action_for_object(obj, action_name):
    """
    Create an Action and bind it to ``obj`` in a way that works across
    Blender 4.3 (legacy actions) and 4.4+/5.x (slotted actions).

    Returns the (action, slot) tuple; ``slot`` is None on legacy Blender.
    """
    action = bpy.data.actions.new(name=action_name)

    if not obj.animation_data:
        obj.animation_data_create()

    anim_data = obj.animation_data

    if not BLENDER_4_4_PLUS:
        # Legacy API: a plain assignment is enough.
        anim_data.action = action
        return action, None

    # Slotted actions: create a slot targeting the object's ID type, assign
    # the action, then explicitly bind the slot so the object is actually
    # animated.
    id_type = obj.id_type  # 'OBJECT' for object.animation_data
    slot = action.slots.new(id_type=id_type, name=obj.name)

    anim_data.action = action

    if getattr(anim_data, "action_slot", None) is None:
        suitable = getattr(anim_data, "action_suitable_slots", None)
        if suitable:
            anim_data.action_slot = suitable[0]
        else:
            anim_data.action_slot = slot

    return action, anim_data.action_slot


def get_action_fcurves(obj):
    """
    Return the F-Curves of the action bound to ``obj``.

    Blender 5.0 removed ``action.fcurves``; the curves now live on the
    channelbag of the assigned action slot.
    """
    anim_data = getattr(obj, "animation_data", None)
    if not anim_data or not anim_data.action:
        return []

    action = anim_data.action

    if not BLENDER_4_4_PLUS:
        return list(action.fcurves)

    try:
        from bpy_extras import anim_utils
    except ImportError:
        return []

    slot = getattr(anim_data, "action_slot", None)
    if slot is None:
        slot = anim_utils.action_get_first_suitable_slot(action, obj.id_type)
    if slot is None:
        return []

    channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
    if channelbag is None:
        return []
    return list(channelbag.fcurves)


# ============================================================================
# Bone colours
# ============================================================================

def set_bone_color(bone, palette='CUSTOM', normal=(0.0, 0.75, 1.0),
                   select=(1.0, 0.85, 0.0), active=(0.2, 1.0, 0.4)):
    """
    Set a bone's display colour.

    Blender 4.0 moved bone colours from bone groups onto the bone itself and
    exposed them through the ``bone.color`` struct (a ``BoneColor``). The
    older ``bone.color_mode`` / ``bone.custom_color`` attributes never existed
    and are silently ignored, so this function prefers ``bone.color`` and only
    falls back to the legacy attributes if it is missing.
    """
    color = getattr(bone, "color", None)

    if color is not None:
        try:
            color.palette = palette
        except (AttributeError, TypeError, ValueError):
            pass
        if palette == 'CUSTOM':
            custom = getattr(color, "custom", None)
            if custom is not None:
                try:
                    custom.normal = normal
                    custom.select = select
                    custom.active = active
                except (AttributeError, TypeError, ValueError):
                    pass
        return True

    # Legacy fallback (pre-4.0 style attributes).
    try:
        if hasattr(bone, 'color_mode'):
            bone.color_mode = palette
        if palette == 'CUSTOM' and hasattr(bone, 'custom_color'):
            bone.custom_color = normal
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def reset_bone_color(bone):
    """Reset a bone's colour back to the Blender default palette."""
    color = getattr(bone, "color", None)
    if color is not None:
        try:
            color.palette = 'DEFAULT'
            return True
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        if hasattr(bone, 'color_mode'):
            bone.color_mode = 'DEFAULT'
    except (AttributeError, TypeError, ValueError):
        return False
    return True


# ============================================================================
# Materials / node trees
# ============================================================================

def ensure_material_node_tree(mat):
    """
    Return the node tree of a material, creating it if needed.

    ``material.use_nodes`` is deprecated in Blender 5.0 (materials always
    have a node tree) and will be removed in 6.0.
    """
    if mat.node_tree is not None:
        return mat.node_tree

    if not BLENDER_5_PLUS:
        mat.use_nodes = True

    return mat.node_tree


def material_uses_nodes(mat):
    """Whether a material has a usable node tree."""
    if mat is None:
        return False
    if BLENDER_5_PLUS:
        return mat.node_tree is not None
    return bool(mat.use_nodes) and mat.node_tree is not None


# ============================================================================
# Mesh / bmesh access
# ============================================================================

def mesh_polygons(mesh):
    """Return the face collection of a Mesh data-block."""
    if hasattr(mesh, "polygons"):
        return mesh.polygons
    return mesh.faces


def mesh_vertices(mesh):
    """Return the vertex collection of a Mesh data-block."""
    if hasattr(mesh, "vertices"):
        return mesh.vertices
    return mesh.verts


def bmesh_faces(bm):
    """
    Return the face collection of a BMesh.

    Blender 4.x introduced ``bm.polygons`` as an alias, but it was removed
    again in Blender 5.x — only ``bm.faces`` is guaranteed to exist. The
    Blender 4.x-era code in this addon used ``bm.polygons`` on 4.x+ and
    therefore crashed with an AttributeError on 5.x.
    """
    faces = getattr(bm, "faces", None)
    if faces is None:
        faces = getattr(bm, "polygons", None)
    return faces


def bmesh_ensure_lookup(bm):
    """
    Build the internal index tables of a BMesh.

    Required before using ``bm.verts[i]`` / ``bm.faces[i]`` indexing. Only
    ``BMElemSeq`` for faces/verts/edges supports ``ensure_lookup_table()``;
    ``bm.loops`` does not.
    """
    for seq in (getattr(bm, "faces", None),
                getattr(bm, "verts", None),
                getattr(bm, "edges", None)):
        if seq is not None and hasattr(seq, "ensure_lookup_table"):
            seq.ensure_lookup_table()


def mesh_uv_layers(mesh):
    """Return the UV layer collection of a Mesh data-block."""
    if hasattr(mesh, "uv_layers"):
        return mesh.uv_layers
    return getattr(mesh, "loops", None) and getattr(mesh.loops, "uv_layers", None)


def get_uv_layer_values(layer, loop_index):
    """
    Read the UV coordinates of a loop from a UV layer.

    Blender 5.0 deprecated ``MeshUVLoopLayer.data`` in favour of the
    ``MeshUVLoopLayer.uv`` collection of ``Float2AttributeValue`` entries.
    Both are supported here.
    """
    if layer is None:
        return (0.0, 0.0)

    uv_coll = getattr(layer, "uv", None)
    if uv_coll is not None:
        try:
            value = uv_coll[loop_index]
            return (value.vector[0], value.vector[1])
        except (AttributeError, TypeError, IndexError, KeyError):
            pass

    data = getattr(layer, "data", None)
    if data is not None:
        try:
            uv = data[loop_index].uv
            return (uv[0], uv[1])
        except (AttributeError, TypeError, IndexError, KeyError):
            pass

    return (0.0, 0.0)


def set_uv_layer_value(layer, loop_index, u, v):
    """Write UV coordinates of a loop, supporting both the 4.x and 5.x APIs."""
    if layer is None:
        return False

    uv_coll = getattr(layer, "uv", None)
    if uv_coll is not None:
        try:
            uv_coll[loop_index].vector = (u, v)
            return True
        except (AttributeError, TypeError, IndexError, KeyError):
            pass

    data = getattr(layer, "data", None)
    if data is not None:
        try:
            data[loop_index].uv = (u, v)
            return True
        except (AttributeError, TypeError, IndexError, KeyError):
            pass

    return False


def mesh_color_layers(mesh):
    """
    Return the corner-domain colour layers of a Mesh data-block.

    Blender 3.2+ stores vertex colours in ``mesh.color_attributes`` instead of
    the legacy ``mesh.vertex_colors`` / ``mesh.loops.vertex_colors``.
    """
    color_attributes = getattr(mesh, "color_attributes", None)
    if color_attributes is None:
        return None

    layers = []
    for attr in color_attributes:
        if getattr(attr, "domain", None) != 'CORNER':
            continue
        if getattr(attr, "data_type", None) not in ('BYTE_COLOR', 'FLOAT_COLOR'):
            continue
        layers.append(attr)
    return layers


def bmesh_color_layer(bm):
    """
    Return the active colour layer of a BMesh, looking through both the
    byte-colour and float-colour layer collections.
    """
    layers = getattr(bm, "loops", None)
    if layers is None:
        return None
    layer_collections = getattr(layers, "layers", None)
    if layer_collections is None:
        return None

    for coll_name in ("color", "float_color"):
        coll = getattr(layer_collections, coll_name, None)
        if coll is None:
            continue
        active = getattr(coll, "active", None)
        if active is not None:
            return active

    return None


def bmesh_deform_layer(bm):
    """Return the active deform (skin) layer of a BMesh, or None."""
    verts = getattr(bm, "verts", None)
    if verts is None:
        return None
    layers = getattr(verts, "layers", None)
    if layers is None:
        return None
    deform = getattr(layers, "deform", None)
    if deform is None:
        return None
    return getattr(deform, "active", None)


def read_deform_weights(bm, vert, deform_layer):
    """
    Read the skin weights of a BMesh vertex.

    Returns a list of ``(vertex_group_index, weight)`` tuples. In Blender 5.x
    the deform data is exposed as a dict-like ``BMDeformVert`` rather than a
    plain sequence.
    """
    if deform_layer is None:
        return []

    try:
        data = vert[deform_layer]
    except (KeyError, TypeError, IndexError):
        return []

    # Blender 5.x: dict-like {group_index: weight}
    if hasattr(data, "items"):
        try:
            return [(int(idx), float(weight)) for idx, weight in data.items()]
        except (TypeError, ValueError):
            return []

    # Blender 4.x: sequence of (group_index, weight) tuples
    try:
        return [(int(idx), float(weight)) for idx, weight in data]
    except (TypeError, ValueError):
        return []
