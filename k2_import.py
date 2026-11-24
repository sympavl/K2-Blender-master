"""
K2 engine model (.model) and animation clip (.clip) importer for Blender.
Supports SMDL v1/v3 meshes and CLIP v1/v2+ animations.

Tested with Blender 4.x and 5.x
"""

import bpy
import struct
import math
import os

try:
    from . import chunk
except ImportError:
    import chunk

from mathutils import Vector, Matrix, Euler

try:
    from .k2_common import (
        log, vlog, dlog, err,
        read_int, read_float,
        bone_depth,
        round_vector, round_matrix,
        vec_roll_to_mat3, mat3_to_vec_roll,
        view_all_in_3d_view,
        MKEY_X, MKEY_Y, MKEY_Z,
        MKEY_PITCH, MKEY_ROLL, MKEY_YAW,
        MKEY_SCALE_X, MKEY_SCALE_Y, MKEY_SCALE_Z,
        MKEY_VISIBILITY,
    )
except ImportError:
    from k2_common import (
        log, vlog, dlog, err,
        read_int, read_float,
        bone_depth,
        round_vector, round_matrix,
        vec_roll_to_mat3, mat3_to_vec_roll,
        view_all_in_3d_view,
        MKEY_X, MKEY_Y, MKEY_Z,
        MKEY_PITCH, MKEY_ROLL, MKEY_YAW,
        MKEY_SCALE_X, MKEY_SCALE_Y, MKEY_SCALE_Z,
        MKEY_VISIBILITY,
    )

# Blender version compatibility helpers (slotted actions, bone colors, ...)
try:
    from .k2_compat import (
        create_action_for_object,
        set_uv_layer_value, get_uv_layer_values,
        BLENDER_VERSION, BLENDER_4_PLUS, BLENDER_5_PLUS,
    )
except ImportError:
    from k2_compat import (
        create_action_for_object,
        set_uv_layer_value, get_uv_layer_values,
        BLENDER_VERSION, BLENDER_4_PLUS, BLENDER_5_PLUS,
    )

# Import texture handling from k2_texture module
try:
    from .k2_texture import (
        _TEX_KEYWORDS,
        assign_textures_to_objects,
        _setup_material_textures,
    )
except ImportError:
    from k2_texture import (
        _TEX_KEYWORDS,
        assign_textures_to_objects,
        _setup_material_textures,
    )

# ============================================================================
# Blender version detection
# ============================================================================
# BLENDER_VERSION / BLENDER_4_PLUS / BLENDER_5_PLUS come from k2_compat.

# ============================================================================
# Chunk parsers — each reads one chunk type from the binary stream
# ============================================================================

def parse_links(honchunk, bone_names):
    mesh_index = read_int(honchunk)
    numverts = read_int(honchunk)
    log("Parsing links")
    vlog(f"Mesh index: {mesh_index}")
    vlog(f"Number of vertices: {numverts}")

    vgroups = {}
    for i in range(numverts):
        num_weights = read_int(honchunk)
        if num_weights > 0:
            weights = struct.unpack(f"<{num_weights}f", honchunk.read(num_weights * 4))
            indexes = struct.unpack(f"<{num_weights}I", honchunk.read(num_weights * 4))
        else:
            weights = indexes = []

        for ii, index in enumerate(indexes):
            name = bone_names[index]
            if name not in vgroups:
                vgroups[name] = []
            vgroups[name].append((i, weights[ii]))

    honchunk.skip()
    return vgroups


def parse_links_single(honchunk, bone_names):
    """Parse single link data (lnk2) - one bone per vertex."""
    mesh_index = read_int(honchunk)
    numverts = read_int(honchunk)
    log("Parsing single links (lnk2)")
    vlog(f"Mesh index: {mesh_index}")
    vlog(f"Number of vertices: {numverts}")

    vgroups = {}
    for i in range(numverts):
        vertex_bone = read_int(honchunk)
        bone_index = read_int(honchunk)
        if bone_index >= 0 and bone_index < len(bone_names):
            name = bone_names[bone_index]
            if name not in vgroups:
                vgroups[name] = []
            vgroups[name].append((i, 1.0))

    honchunk.skip()
    return vgroups


def parse_vertices(honchunk):
    vlog('Parsing vertices chunk')
    numverts = int((honchunk.chunksize - 4) / 12)
    vlog(f'{numverts} vertices')
    meshindex = read_int(honchunk)
    return [struct.unpack("<3f", honchunk.read(12)) for _ in range(numverts)]


def parse_sign(honchunk):
    vlog('Parsing sign chunk')
    numverts = honchunk.chunksize - 8
    meshindex = read_int(honchunk)
    read_int(honchunk)  # padding / unknown field
    return [struct.unpack("<b", honchunk.read(1)) for _ in range(numverts)]


def parse_faces(honchunk, version):
    vlog('Parsing faces chunk')
    meshindex = read_int(honchunk)
    numfaces = read_int(honchunk)
    vlog(f'{numfaces} faces')

    if version == 3:
        size = struct.unpack('B', honchunk.read(1))[0]
    elif version == 1:
        size = 4

    fmt_map = {1: ("<3B", 3), 2: ("<3H", 6), 4: ("<3I", 12)}
    if size in fmt_map:
        fmt, nbytes = fmt_map[size]
        return [struct.unpack(fmt, honchunk.read(nbytes)) for _ in range(numfaces)]

    log(f"Unknown size for faces: {size}")
    return []


def parse_normals(honchunk):
    vlog('Parsing normals chunk')
    numverts = int((honchunk.chunksize - 4) / 12)
    vlog(f'{numverts} normals')
    meshindex = read_int(honchunk)
    return [struct.unpack("<3f", honchunk.read(12)) for _ in range(numverts)]


def parse_texc(honchunk, version):
    vlog('Parsing UV texc chunk')
    meshindex = read_int(honchunk)
    uv_channel = 0
    if version == 3:
        uv_channel = read_int(honchunk)
        numverts = (honchunk.chunksize - 8) // 8
    else:
        numverts = (honchunk.chunksize - 4) // 8
    vlog(f'{numverts} texc (channel {uv_channel})')
    uvs = [struct.unpack("<2f", honchunk.read(8)) for _ in range(numverts)]
    return uv_channel, uvs


def parse_colr(honchunk):
    vlog('Parsing vertex colors chunk')
    numverts = int((honchunk.chunksize - 4) / 4)
    meshindex = read_int(honchunk)
    return [struct.unpack("<4B", honchunk.read(4)) for _ in range(numverts)]


def parse_surf(honchunk):
    vlog('Parsing surface chunk')
    surfindex = read_int(honchunk)
    num_planes = read_int(honchunk)
    num_points = read_int(honchunk)
    num_edges = read_int(honchunk)
    num_tris = read_int(honchunk)

    # BMIN(3f), BMAX(3f), FLAGS(i)
    honchunk.read(4 * 3 + 4 * 3 + 4)
    return (
        [struct.unpack("<4f", honchunk.read(4 * 4)) for _ in range(num_planes)],
        [struct.unpack("<3f", honchunk.read(4 * 3)) for _ in range(num_points)],
        [struct.unpack("<6f", honchunk.read(4 * 6)) for _ in range(num_edges)],
        [struct.unpack("<3I", honchunk.read(4 * 3)) for _ in range(num_tris)],
    )


# ============================================================================
# Sub-chunk dispatcher — reads child chunks inside a mesh block
# ============================================================================

def _read_mesh_subchunks(file, version, bone_names):
    """Read all sub-chunks that follow a 'mesh' chunk until the next mesh/surf or EOF."""
    verts = []
    faces = []
    signs = []
    nrml = []
    texc = []
    colors = []
    vgroups = {}
    honchunk = None

    while True:
        try:
            honchunk = chunk.Chunk(file, bigendian=False, align=False)
        except EOFError:
            vlog('Done reading chunks')
            return verts, faces, signs, nrml, texc, colors, vgroups, None

        name = honchunk.getname()
        if name in (b'mesh', b'surf'):
            return verts, faces, signs, nrml, texc, colors, vgroups, honchunk

        if name == b'vrts':
            verts = parse_vertices(honchunk)
        elif name == b'face':
            faces = parse_faces(honchunk, version)
        elif name == b'nrml':
            nrml = parse_normals(honchunk)
        elif name == b'texc':
            channel, uvs = parse_texc(honchunk, version)
            if channel == 0 or not texc:
                texc = uvs
            else:
                vlog(f'Skipping UV channel {channel} (using channel 0)')
        elif name == b'colr':
            colors = parse_colr(honchunk)
        elif name in (b'lnk1', b'lnk3'):
            vgroups = parse_links(honchunk, bone_names)
        elif name == b'lnk2':
            vgroups = parse_links_single(honchunk, bone_names)
        elif name == b'sign':
            signs = parse_sign(honchunk)
        else:
            vlog(f'Skipping chunk: {name}')

        honchunk.skip()


# ============================================================================
# Bone reading
# ============================================================================

def _read_bone_v3(honchunk):
    """Read a single bone entry in SMDL version 3 format."""
    parent_index = read_int(honchunk)
    inv_matrix = Matrix([struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                         struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                         struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                         struct.unpack('<3f', honchunk.read(12)) + (1.0,)])
    matrix = Matrix([struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                     struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                     struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                     struct.unpack('<3f', honchunk.read(12)) + (1.0,)])
    name_length = struct.unpack("B", honchunk.read(1))[0]
    name = honchunk.read(name_length).decode()
    honchunk.read(1)  # null terminator
    return parent_index, name, matrix


def _read_bone_v1(honchunk):
    """Read a single bone entry in SMDL version 1 format."""
    parent_index = read_int(honchunk)
    pos = honchunk.tell() - 4
    b = honchunk.read(1)
    name = ''
    while b != b'\0':
        name += b.decode()
        b = honchunk.read(1)
    honchunk.seek(pos + 0x24)
    inv_matrix = Matrix([struct.unpack('<4f', honchunk.read(16)),
                         struct.unpack('<4f', honchunk.read(16)),
                         struct.unpack('<4f', honchunk.read(16)),
                         struct.unpack('<4f', honchunk.read(16))])
    matrix = Matrix([struct.unpack('<4f', honchunk.read(16)),
                     struct.unpack('<4f', honchunk.read(16)),
                     struct.unpack('<4f', honchunk.read(16)),
                     struct.unpack('<4f', honchunk.read(16))])
    return parent_index, name, matrix


def _read_bones(honchunk, num_bones, version, armature_data):
    """Read all bones, create edit bones, and return bone_names + parent indices."""
    bones = []
    bone_names = []
    parents = []

    for _ in range(num_bones):
        if version == 3:
            parent_index, name, matrix = _read_bone_v3(honchunk)
        elif version == 1:
            parent_index, name, matrix = _read_bone_v1(honchunk)
        else:
            err(f"Unsupported version {version}")
            return [], [], []

        log(f"Bone name: {name}, parent {parent_index}")
        bone_names.append(name)

        matrix.transpose()
        matrix = round_matrix(matrix, 4)
        pos = matrix.translation
        axis, roll = mat3_to_vec_roll(matrix.to_3x3())

        bone = armature_data.edit_bones.new(name)
        bone.head = pos
        bone.tail = pos + axis
        bone.roll = roll
        parents.append(parent_index)
        bones.append(bone)

    for i in range(num_bones):
        if parents[i] != -1:
            bones[i].parent = bones[parents[i]]

    honchunk.skip()
    return bones, bone_names, parents


# ============================================================================
# Mesh header reading
# ============================================================================

def _read_mesh_header_v3(honchunk):
    """Read mesh header fields for SMDL v3."""
    mode = read_int(honchunk)
    vlog(f"Mode: {mode}")
    vlog(f"Vertices count: {read_int(honchunk)}")
    vlog("Bounding box: (%f, %f, %f) - (%f, %f, %f)" % struct.unpack("<ffffff", honchunk.read(24)))
    bone_link = read_int(honchunk)
    vlog(f"Bone link: {bone_link}")
    sizename = struct.unpack('B', honchunk.read(1))[0]
    sizemat = struct.unpack('B', honchunk.read(1))[0]
    meshname = honchunk.read(sizename).decode().rstrip('\x00')
    honchunk.read(1)  # null terminator
    materialname = honchunk.read(sizemat).decode().rstrip('\x00')
    return mode, bone_link, meshname, materialname


def _read_mesh_header_v1(honchunk):
    """Read mesh header fields for SMDL v1."""
    pos = honchunk.tell() - 4
    b = honchunk.read(1)
    meshname = ''
    while b != b'\0':
        meshname += b.decode()
        b = honchunk.read(1)
    honchunk.seek(pos + 0x24)
    b = honchunk.read(1)
    materialname = ''
    while b != b'\0':
        materialname += b.decode()
        b = honchunk.read(1)
    return 1, -1, meshname, materialname


# ============================================================================
# Blender object builders
# ============================================================================

def _build_mesh_object(scn, meshname, materialname, verts, faces, texc, flipuv,
                       vgroups, bone_link, bone_names, rig, is_surf,
                       model_dir='', tex_flags=None):
    """Create a Blender mesh object with UVs, vertex groups, and armature modifier."""
    msh = bpy.data.meshes.new(name=meshname)
    msh.from_pydata(verts, [], faces)
    msh.update()

    if materialname:
        mat = bpy.data.materials.new(materialname)
        msh.materials.append(mat)
        if model_dir and tex_flags:
            _setup_material_textures(mat, model_dir, materialname, tex_flags)

    if texc:
        if flipuv:
            texc = [(uv[0], 1 - uv[1]) for uv in texc]
        # Blender 5.0 deprecated `uv_layer.data` in favour of `uv_layer.uv`.
        uv_layer = msh.uv_layers.new(name=f'UVMain{meshname}')
        for face in msh.polygons:
            for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
                set_uv_layer_value(uv_layer, loop_idx, texc[vert_idx][0], texc[vert_idx][1])

    obj = bpy.data.objects.new(f'{meshname}_Object', msh)
    scn.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()

    if is_surf:
        obj.display_type = 'WIRE'
    else:
        # Reuse vertex groups by name: the bone-link group and the skin
        # weight groups can refer to the same bone, and calling
        # `vertex_groups.new()` twice with the same name produces a stray
        # "BoneName.001" duplicate group.
        groups = {}

        def _get_group(name):
            grp = groups.get(name)
            if grp is None:
                grp = obj.vertex_groups.get(name)
                if grp is None:
                    grp = obj.vertex_groups.new(name=name)
                groups[name] = grp
            return grp

        if bone_link >= 0 and bone_link < len(bone_names):
            grp = _get_group(bone_names[bone_link])
            grp.add(list(range(len(msh.vertices))), 1.0, 'REPLACE')
        for name, vg in vgroups.items():
            grp = _get_group(name)
            for v, w in vg:
                grp.add([v], w, 'REPLACE')

        mod = obj.modifiers.new(name='K2_Armature', type='ARMATURE')
        mod.object = rig
        mod.use_bone_envelopes = False
        mod.use_vertex_groups = True

        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')
        for b in rig.pose.bones:
            b.rotation_mode = 'QUATERNION'
        bpy.ops.object.mode_set(mode='OBJECT')
        rig.select_set(False)

    bpy.context.view_layer.objects.active = None
    return obj


# ============================================================================
# Main mesh import
# ============================================================================

def create_blender_mesh(filename, objname, flipuv, tex_flags=None):
    """Import a K2 .model file into Blender."""
    import sys
    
    model_dir = os.path.dirname(os.path.abspath(filename))
    log(f"Starting import of: {filename}")
    log(f"File size: {os.path.getsize(filename)} bytes")
    
    try:
        with open(filename, 'rb') as file:
            sig = file.read(4)
            log(f"File signature: {sig}")
            if sig != b'SMDL':
                err('Unknown file signature')
                return None, None

            honchunk = chunk.Chunk(file, bigendian=False, align=False)
            log(f"First chunk: {honchunk.getname()}")
            if honchunk.getname() != b'head':
                err('File does not start with head chunk')
                return None, None

            version = read_int(honchunk)
            num_meshes = read_int(honchunk)
            num_sprites = read_int(honchunk)
            num_surfs = read_int(honchunk)
            num_bones = read_int(honchunk)

            log(f"K2 Model v{version}: {num_meshes} mesh(es), {num_sprites} sprite(s), {num_surfs} surf(s), {num_bones} bone(s)")
            if num_meshes == 0:
                log("WARNING: Model file contains 0 meshes — only bones will be imported")
            vlog("Bounding box: (%f, %f, %f) - (%f, %f, %f)" % struct.unpack("<ffffff", honchunk.read(24)))
            honchunk.skip()

            scn = bpy.context.scene

            # --- Read bones ---
            try:
                honchunk = chunk.Chunk(file, bigendian=False, align=False)
            except EOFError:
                err('Error reading bone chunk')
                return None, None

            armature_data = bpy.data.armatures.new(f'{objname}_Armature')
            armature_data.display_type = 'STICK'
            armature_data.show_names = True
            rig = bpy.data.objects.new(f'{objname}_Rig', armature_data)
            scn.collection.objects.link(rig)
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)

            bpy.ops.object.mode_set(mode='EDIT')
            _, bone_names, _ = _read_bones(honchunk, num_bones, version, armature_data)
            bpy.ops.object.mode_set(mode='OBJECT')

            rig.show_in_front = True
            bpy.context.view_layer.update()
            log(f"Created armature with {len(bone_names)} bones")

            # --- Read meshes / surfs ---
            try:
                honchunk = chunk.Chunk(file, bigendian=False, align=False)
            except EOFError:
                err('Error reading mesh chunk')
                return None, rig

            obj = None
            mesh_count = 0
            while honchunk and honchunk.getname() in (b'mesh', b'surf'):
                if honchunk.getname() == b'mesh':
                    vlog(f"Mesh index: {read_int(honchunk)}")

                    if version == 3:
                        mode, bone_link, meshname, materialname = _read_mesh_header_v3(honchunk)
                    elif version == 1:
                        mode, bone_link, meshname, materialname = _read_mesh_header_v1(honchunk)
                    else:
                        err(f"Unsupported version {version}")
                        return obj, rig

                    honchunk.skip()

                    if mode == 3 or mode == 4:
                        # Sprite modes: 3=billboard, 4=ground plane
                        sprite_type = 'SPRITE' if mode == 3 else 'GROUND'
                        vlog(f"Importing sprite: {meshname} (type={sprite_type})")

                        # Read sprite mesh data
                        verts, faces, signs, nrml, texc, colors, vgroups, honchunk = \
                            _read_mesh_subchunks(file, version, bone_names)

                        obj = _build_mesh_object(
                            scn, meshname, materialname, verts, faces, texc, flipuv,
                            {}, -1, bone_names, rig, is_surf=False,
                            model_dir=model_dir, tex_flags=tex_flags,
                        )
                        if obj:
                            log(f"Created sprite object: {obj.name}")
                            mesh_count += 1
                        # Mark as sprite for identification
                        if hasattr(obj, 'k2_mesh_settings'):
                            obj.k2_mesh_settings.mesh_type = sprite_type

                    elif mode != 1:
                        # Skip other non-standard mesh modes
                        while True:
                            try:
                                honchunk = chunk.Chunk(file, bigendian=False, align=False)
                            except EOFError:
                                honchunk = None
                                break
                            if honchunk.getname() in (b'mesh', b'surf'):
                                break
                            honchunk.skip()
                        continue

                    verts, faces, signs, nrml, texc, colors, vgroups, honchunk = \
                        _read_mesh_subchunks(file, version, bone_names)

                    log(f"Parsed mesh '{meshname}': {len(verts)} verts, {len(faces)} faces")
                    
                    if not verts or not faces:
                        log(f"WARNING: Mesh '{meshname}' has no geometry, skipping")
                        # Skip to next chunk
                        while True:
                            try:
                                honchunk = chunk.Chunk(file, bigendian=False, align=False)
                            except EOFError:
                                honchunk = None
                                break
                            if honchunk.getname() in (b'mesh', b'surf'):
                                break
                            honchunk.skip()
                        continue
                    
                    obj = _build_mesh_object(
                        scn, meshname, materialname, verts, faces, texc, flipuv,
                        vgroups, bone_link, bone_names, rig, is_surf=False,
                        model_dir=model_dir, tex_flags=tex_flags,
                    )
                    
                    if obj:
                        log(f"Created object: {obj.name}")
                        mesh_count += 1

                elif honchunk.getname() == b'surf':
                    surf_planes, surf_points, surf_edges, surf_tris = parse_surf(honchunk)
                    dlog(f"Surface: {len(surf_planes)} planes, {len(surf_points)} points, "
                         f"{len(surf_edges)} edges, {len(surf_tris)} tris")

                    honchunk.skip()

                    obj = _build_mesh_object(
                        scn, f'{objname}_surf', None, surf_points, surf_tris, [], flipuv,
                        {}, -1, bone_names, rig, is_surf=True,
                        model_dir=model_dir, tex_flags=None,
                    )

                    try:
                        honchunk = chunk.Chunk(file, bigendian=False, align=False)
                    except EOFError:
                        vlog('Done reading chunks')
                        honchunk = None

            log(f"Import complete: {mesh_count} mesh(es) created")
            log(f"Total objects in scene: {len(list(bpy.data.objects))}")
            
            bpy.context.view_layer.update()
            view_all_in_3d_view()
            return obj, rig

    except IOError as e:
        err(f"File IO Error: {e}")
    except Exception as e:
        err(f"Unexpected error: {e}")
    return None, None


# ============================================================================
# Clip helpers
# ============================================================================

def _get_motion_value(motion_keys, frame_index):
    """Safely get a motion key value, clamping to the last available frame."""
    if frame_index < len(motion_keys):
        return motion_keys[frame_index]
    return motion_keys[-1]


def get_transform_matrix(motions, bone, frame, version):
    motion = motions[bone.name]
    x = _get_motion_value(motion[MKEY_X], frame)
    y = _get_motion_value(motion[MKEY_Y], frame)
    z = _get_motion_value(motion[MKEY_Z], frame)

    rx = _get_motion_value(motion[MKEY_PITCH], frame)
    ry = _get_motion_value(motion[MKEY_ROLL], frame)
    rz = _get_motion_value(motion[MKEY_YAW], frame)

    sx = _get_motion_value(motion[MKEY_SCALE_X], frame)
    if version == 1:
        sy = sz = sx
    else:
        sy = _get_motion_value(motion[MKEY_SCALE_Y], frame)
        sz = _get_motion_value(motion[MKEY_SCALE_Z], frame)

    rotation_matrix = Euler(
        (math.radians(rx), math.radians(ry), math.radians(rz)), 'YXZ'
    ).to_matrix().to_4x4()
    transform = Matrix.Translation(Vector((x, y, z))) @ rotation_matrix

    return transform, Vector((sx, sy, sz))


def animate_bone(name, pose, motions, num_frames, armature, arm_ob, version):
    if armature is None or not hasattr(armature, 'bones'):
        err(f"Cannot animate '{name}': no armature data")
        return

    if name not in armature.bones.keys():
        log(f'{name} not found in armature')
        return

    bone = armature.bones[name]
    bone_rest_matrix = Matrix(bone.matrix_local)

    if bone.parent is not None:
        parent_rest_inv = Matrix(bone.parent.matrix_local)
        parent_rest_inv.invert()
        bone_rest_matrix = parent_rest_inv @ bone_rest_matrix

    bone_rest_matrix_inv = Matrix(bone_rest_matrix).inverted()

    pbone = pose.bones[name]
    for i in range(num_frames):
        transform, size = get_transform_matrix(motions, bone, i, version)
        transform = bone_rest_matrix_inv @ transform
        pbone.rotation_quaternion = transform.to_quaternion()
        pbone.location = transform.to_translation()
        pbone.keyframe_insert(data_path='rotation_quaternion', frame=i)
        pbone.keyframe_insert(data_path='location', frame=i)


# ============================================================================
# Main clip import
# ============================================================================

def _find_target_armature(clip_bone_names=None):
    """
    Find the armature a clip should be applied to.

    The user often has both the imported mesh and the rig selected. Picking
    ``selected_objects[0]`` blindly then yields the mesh, and the clip import
    dies with "'Mesh' object has no attribute 'bones'".

    Preference order:
      1. the active object, if it is an armature
      2. the only selected armature, if exactly one is selected
      3. the first selected armature otherwise
      4. the armature whose bones best match the clip, if it is unambiguous
      5. the only armature in the scene
    """
    view_layer = bpy.context.view_layer

    selected = [o for o in view_layer.objects
                if o.select_get() and o.type == 'ARMATURE']

    # The active object wins when it is an armature.
    active = view_layer.objects.active
    if active is not None and active.type == 'ARMATURE':
        return active

    if selected:
        return selected[0]

    armatures = [o for o in view_layer.objects if o.type == 'ARMATURE']
    if not armatures:
        return None

    # Nothing selected: pick the armature whose bones match the clip best.
    if clip_bone_names:
        wanted = set(clip_bone_names)
        scored = []
        for arm_ob in armatures:
            bone_names = set(arm_ob.data.bones.keys())
            matched = wanted & bone_names
            if not matched:
                continue
            # Coverage of the clip's bones matters more than the raw count:
            # a rig with all the clip's bones is a better target than a rig
            # that merely owns a lot of bones with the same names.
            coverage = len(matched) / max(len(wanted), 1)
            scored.append((coverage, len(matched), arm_ob))

        if scored:
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            top_coverage, top_count, best = scored[0]
            tie = len(scored) > 1 and (
                scored[1][0] == top_coverage and scored[1][1] == top_count)
            if not tie:
                log(f"Using armature '{best.name}' "
                    f"(matches {top_count} of {len(wanted)} clip bones)")
                return best
            log(f"Ambiguous: several armatures match the clip "
                f"({', '.join(o.name for _, _, o in scored)}) - select one")

    if len(armatures) == 1:
        log(f"Using the only armature in the scene: {armatures[0].name}")
        return armatures[0]

    return None


def create_blender_clip(filename, clipname):
    """
    Import a K2 .clip file and apply it to the selected armature.

    Returns True on success, False on failure.
    """
    try:
        with open(filename, 'rb') as file:
            sig = file.read(4)
            if sig != b'CLIP':
                err('Unknown file signature')
                return False

            clipchunk = chunk.Chunk(file, bigendian=False, align=False)
            version = read_int(clipchunk)
            num_bones = read_int(clipchunk)
            num_frames = read_int(clipchunk)
            vlog(f"Version: {version}")
            vlog(f"Number of bones: {num_bones}")
            vlog(f"Number of frames: {num_frames}")

            if not bpy.context.selected_objects and not [
                    o for o in bpy.context.view_layer.objects if o.type == 'ARMATURE']:
                err('No object selected')
                return False

            motions = {}

            while True:
                try:
                    clipchunk = chunk.Chunk(file, bigendian=False, align=False)
                except EOFError:
                    break

                if version == 1:
                    name = clipchunk.read(32).split(b'\0', 1)[0]
                boneindex = read_int(clipchunk)
                keytype = read_int(clipchunk)
                numkeys = read_int(clipchunk)
                if version > 1:
                    namelength = struct.unpack("B", clipchunk.read(1))[0]
                    name = clipchunk.read(namelength)
                    clipchunk.read(1)
                name = name.decode("utf8")

                if name not in motions:
                    motions[name] = {}
                dlog(f"{name}, bone index: {boneindex}, key type: {keytype}, keys: {numkeys}")
                if keytype == MKEY_VISIBILITY:
                    data = struct.unpack(f"{numkeys}B", clipchunk.read(numkeys))
                else:
                    data = struct.unpack(f"<{numkeys}f", clipchunk.read(numkeys * 4))
                motions[name][keytype] = list(data)
                clipchunk.skip()

            # The armature to animate may not be the first selected object:
            # importing a model leaves the mesh selected next to the rig, and
            # `selected_objects[0].data` would then be a Mesh instead of an
            # Armature ("'Mesh' object has no attribute 'bones'").
            arm_ob = _find_target_armature(motions.keys())
            if arm_ob is None:
                err('Select an armature to import the clip onto')
                return False

            armature = arm_ob.data
            action, slot = create_action_for_object(arm_ob, clipname)
            log(f"Created action '{action.name}' (slot: {slot.identifier if slot else 'legacy'})")
            pose = arm_ob.pose

            for bone_name in motions:
                animate_bone(bone_name, pose, motions, num_frames, armature, arm_ob, version)

            return True

    except IOError as e:
        err(f"File IO Error: {e}")
        return False


# ============================================================================
# Public entry points (called from __init__.py operators)
# ============================================================================

def readclip(filepath):
    obj_name = bpy.path.display_name_from_filepath(filepath)
    return create_blender_clip(filepath, obj_name)


def read(filepath, flipuv, tex_flags=None):
    obj_name = bpy.path.display_name_from_filepath(filepath)
    create_blender_mesh(filepath, obj_name, flipuv, tex_flags=tex_flags)
