package com.example.voxelizer;

import de.javagl.jgltf.model.GltfModel;
import de.javagl.jgltf.model.MeshModel;
import de.javagl.jgltf.model.MeshPrimitiveModel;
import de.javagl.jgltf.model.NodeModel;
import de.javagl.jgltf.model.SceneModel;
import de.javagl.jgltf.model.AccessorByteData;
import de.javagl.jgltf.model.AccessorData;
import de.javagl.jgltf.model.AccessorDatas;
import de.javagl.jgltf.model.AccessorFloatData;
import de.javagl.jgltf.model.AccessorIntData;
import de.javagl.jgltf.model.AccessorModel;
import de.javagl.jgltf.model.AccessorShortData;
import de.javagl.jgltf.model.io.GltfModelReader;
import de.javagl.jgltf.model.io.*;
import de.javagl.jgltf.model.structure.*;
import de.javagl.jgltf.model.AccessorData;
import de.javagl.jgltf.model.AccessorDatas;
import de.javagl.jgltf.model.AccessorFloatData;
import de.javagl.jgltf.model.AccessorByteData;
import de.javagl.jgltf.model.AccessorShortData;
import de.javagl.jgltf.model.AccessorIntData;

import java.nio.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

public final class JavaCpuVoxelizer {

    private final int grid;
    private final boolean tiles3d;
    private final boolean verbose;

    // Hardcoded 3D Tiles cube (your CUDA path uses this)
    private static final float TILE_SCALE = 1.3f;
    private static final float TILE_X = 46.0f * TILE_SCALE;
    private static final float TILE_Y = 38.0f * TILE_SCALE;
    private static final float TILE_Z = 24.0f * TILE_SCALE;

    public JavaCpuVoxelizer(int gridSize, boolean tiles3d, boolean verbose) {
        this.grid = Math.max(8, gridSize);
        this.tiles3d = tiles3d;
        this.verbose = verbose;
    }

    public static final class Stats {
        public final String baseName;
        public final int grid;
        public final int triangles;
        public final int filled;
        public final int ox, oy, oz;
        Stats(String baseName, int grid, int triangles, int filled, int ox, int oy, int oz) {
            this.baseName = baseName; this.grid = grid; this.triangles = triangles; this.filled = filled;
            this.ox = ox; this.oy = oy; this.oz = oz;
        }
    }

    // ------------------------------- Public API -------------------------------

    public Stats voxelizeSingleGLB(File glbFile, File outDir) throws Exception {
        if (verbose) System.out.println("[Load] " + glbFile.getAbsolutePath());

        Mesh mesh = GltfReader.loadTriangles(glbFile);
        if (mesh.tris.isEmpty()) throw new IllegalStateException("No triangles in " + glbFile);

        // Object-space bbox
        Aabb bbox = mesh.computeBounds();
        if (tiles3d) {
            Vec3 c = bbox.center();
            Vec3 half = new Vec3(TILE_X * 0.5f, TILE_Y * 0.5f, TILE_Z * 0.5f);
            bbox = new Aabb(c.sub(half), c.add(half));
            if (verbose) System.out.printf("[BBox] 3DTiles cube   min=(%.3f,%.3f,%.3f)  max=(%.3f,%.3f,%.3f)%n",
                    bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
        } else if (verbose) {
            System.out.printf("[BBox] Mesh bbox       min=(%.3f,%.3f,%.3f)  max=(%.3f,%.3f,%.3f)%n",
                    bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
        }

        // Uniform voxel size from the cube’s max dimension
        Vec3 size = bbox.extent();
        float unit = Math.max(Math.max(size.x, size.y), size.z) / grid;
        if (verbose) System.out.printf("[Grid] %d^3  unit=%.6f%n", grid, unit);

        // Offsets like CUDA JSON writer (ox,oy,oz are rounded rebase to 0)
        int ox = Math.round((-bbox.min.x) / unit);
        int oy = Math.round((-bbox.min.y) / unit);
        int oz = Math.round((-bbox.min.z) / unit);

        // Transform all vertices to voxel space once
        for (Tri t : mesh.tris) {
            t.v0 = t.v0.sub(bbox.min).div(unit);
            t.v1 = t.v1.sub(bbox.min).div(unit);
            t.v2 = t.v2.sub(bbox.min).div(unit);
        }

        // Occupancy
        BitSet occ = new BitSet(grid * grid * grid);

        long t0 = System.nanoTime();
        int triCount = mesh.tris.size();

        // Rasterize each triangle into the grid using a robust tri-box SAT
        for (Tri t : mesh.tris) {
            float minx = Math.min(t.v0.x, Math.min(t.v1.x, t.v2.x));
            float miny = Math.min(t.v0.y, Math.min(t.v1.y, t.v2.y));
            float minz = Math.min(t.v0.z, Math.min(t.v1.z, t.v2.z));
            float maxx = Math.max(t.v0.x, Math.max(t.v1.x, t.v2.x));
            float maxy = Math.max(t.v0.y, Math.max(t.v1.y, t.v2.y));
            float maxz = Math.max(t.v0.z, Math.max(t.v1.z, t.v2.z));

            int x0 = clamp((int)Math.floor(minx), 0, grid - 1);
            int y0 = clamp((int)Math.floor(miny), 0, grid - 1);
            int z0 = clamp((int)Math.floor(minz), 0, grid - 1);
            int x1 = clamp((int)Math.ceil (maxx), 0, grid - 1);
            int y1 = clamp((int)Math.ceil (maxy), 0, grid - 1);
            int z1 = clamp((int)Math.ceil (maxz), 0, grid - 1);

            for (int z = z0; z <= z1; z++) {
                for (int y = y0; y <= y1; y++) {
                    for (int x = x0; x <= x1; x++) {
                        if (!overlapsTriangle(x, y, z, t)) continue;
                        occ.set(x + grid * (y + grid * z));
                    }
                }
            }
        }
        long t1 = System.nanoTime();
        int filled = occ.cardinality();
        if (verbose) System.out.printf("[Raster] tris=%d  filled=%d  time=%.1f ms%n",
                triCount, filled, (t1 - t0)/1e6);

        // Output JSON: one white palette entry "1"
        String base = baseName(glbFile);
        File jsonOut = new File(outDir, base + "_" + grid + ".json");
        JSONObject blocks = new JSONObject();
        blocks.put("1", new JSONArray(new int[]{255,255,255}));

        JSONArray xyzi = new JSONArray();
        for (int z = 0; z < grid; z++) {
            for (int y = 0; y < grid; y++) {
                for (int x = 0; x < grid; x++) {
                    int idx = x + grid * (y + grid * z);
                    if (occ.get(idx)) {
                        xyzi.put(new JSONArray(new int[]{ x - ox, y - oy, z - oz, 1 }));
                    }
                }
            }
        }
        JSONObject root = new JSONObject();
        root.put("blocks", blocks);
        root.put("xyzi", xyzi);
        try (FileWriter fw = new FileWriter(jsonOut)) {
            fw.write(root.toString());
        }
        if (verbose) System.out.println("[Write] " + jsonOut.getName() + " (" + xyzi.length() + " voxels)");

        // Output position file
        File posOut = new File(outDir, base + "_position.json");
        JSONArray arr = new JSONArray();
        JSONObject pos = new JSONObject();
        pos.put("translation", new JSONArray(new int[]{ox, oy, oz}));
        pos.put("origin", new JSONArray(new int[]{0, 0, 0}));
        arr.put(pos);
        try (FileWriter fw = new FileWriter(posOut)) {
            fw.write(arr.toString());
        }
        if (verbose) System.out.println("[Write] " + posOut.getName() +
                "  translation=["+ox+","+oy+","+oz+"] origin=[0,0,0]");

        return new Stats(base, grid, triCount, filled, ox, oy, oz);
    }

    // ------------------------------- Geometry types -------------------------------

    private static final class Vec3 {
        final float x, y, z;
        Vec3(float x, float y, float z){ this.x=x; this.y=y; this.z=z; }
        Vec3 add(Vec3 o){ return new Vec3(x+o.x,y+o.y,z+o.z); }
        Vec3 sub(Vec3 o){ return new Vec3(x-o.x,y-o.y,z-o.z); }
        Vec3 mul(float s){ return new Vec3(x*s,y*s,z*s); }
        Vec3 div(float s){ return new Vec3(x/s,y/s,z/s); }
        Vec3 cross(Vec3 o){ return new Vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x); }
    }
    private static final class Aabb {
        final Vec3 min, max;
        Aabb(Vec3 min, Vec3 max){ this.min=min; this.max=max; }
        Vec3 extent(){ return new Vec3(max.x-min.x, max.y-min.y, max.z-min.z); }
        Vec3 center(){ return new Vec3((min.x+max.x)/2f,(min.y+max.y)/2f,(min.z+max.z)/2f); }
    }
    private static final class Tri {
        Vec3 v0, v1, v2;
    }
    private static final class Mesh {
        final List<Tri> tris = new ArrayList<>();
        Aabb computeBounds() {
            float minx=Float.POSITIVE_INFINITY,miny=Float.POSITIVE_INFINITY,minz=Float.POSITIVE_INFINITY;
            float maxx=Float.NEGATIVE_INFINITY,maxy=Float.NEGATIVE_INFINITY,maxz=Float.NEGATIVE_INFINITY;
            for (Tri t : tris) {
                for (Vec3 v : new Vec3[]{t.v0,t.v1,t.v2}) {
                    minx = Math.min(minx, v.x); miny = Math.min(miny, v.y); minz = Math.min(minz, v.z);
                    maxx = Math.max(maxx, v.x); maxy = Math.max(maxy, v.y); maxz = Math.max(maxz, v.z);
                }
            }
            return new Aabb(new Vec3(minx,miny,minz), new Vec3(maxx,maxy,maxz));
        }
    }

    // ------------------------------- GLB reader -------------------------------

private static final class GltfReader {
    static Mesh loadTriangles(File glbFile) throws Exception {
        byte[] data = java.nio.file.Files.readAllBytes(glbFile.toPath());
        GltfModel model = new GltfModelReader().readWithoutReferences(new ByteArrayInputStream(data));

        Mesh out = new Mesh();

        // Walk scenes → nodes (works across 2.0.x)
        for (SceneModel scene : model.getSceneModels()) {
            for (NodeModel root : scene.getNodeModels()) {
                traverseNode(root, out);
            }
        }
        return out;
    }

private static void traverseNode(NodeModel node, Mesh out) {
    // glTF 2.0: node → 0..n meshes
    List<MeshModel> meshes = node.getMeshModels();  // <-- was getMeshModel()
    if (meshes != null) {
        for (MeshModel meshModel : meshes) {
            for (MeshPrimitiveModel prim : meshModel.getMeshPrimitiveModels()) {
                addPrimitive(prim, out);
            }
        }
    }
    for (NodeModel child : node.getChildren()) {
        traverseNode(child, out);
    }
}


    private static void addPrimitive(MeshPrimitiveModel pm, Mesh out) {
        AccessorModel posAcc = pm.getAttributes().get("POSITION");
        if (posAcc == null) return;

        // Typed float accessor for positions
        AccessorFloatData pos = AccessorDatas.createFloat(posAcc);

        IntBuffer indices = toIndexBuffer(pm);

        if (indices != null) {
            // indexed
            while (indices.hasRemaining()) {
                int i0 = indices.get();
                int i1 = indices.get();
                int i2 = indices.get();
                out.tris.add(triFrom(pos, i0, i1, i2));
            }
        } else {
            // non-indexed: triangles are sequential
            int count = pos.getNumElements();
            for (int i = 0; i + 2 < count; i += 3) {
                out.tris.add(triFrom(pos, i, i + 1, i + 2));
            }
        }
    }

    private static Tri triFrom(AccessorFloatData pos, int i0, int i1, int i2) {
        Tri t = new Tri();
        t.v0 = readVec3(pos, i0);
        t.v1 = readVec3(pos, i1);
        t.v2 = readVec3(pos, i2);
        return t;
    }

    private static Vec3 readVec3(AccessorFloatData pos, int i) {
        return new Vec3(pos.get(i, 0), pos.get(i, 1), pos.get(i, 2));
    }

    /** Build an IntBuffer from whatever index type the primitive uses (u8/u16/u32). */
/** Build an IntBuffer from whatever index type the primitive uses (u8/u16/u32). */
private static IntBuffer toIndexBuffer(MeshPrimitiveModel pm) {
    AccessorModel idxAcc = pm.getIndices();
    if (idxAcc == null) return null;

    AccessorData ad = AccessorDatas.create(idxAcc); // typed instance behind the interface
    int n = ad.getNumElements();
    IntBuffer ib = IntBuffer.allocate(n);

    if (ad instanceof AccessorByteData bd) {
        // GL_UNSIGNED_BYTE indices may come through here -> mask to 0..255
        for (int i = 0; i < n; i++) ib.put(bd.get(i, 0) & 0xFF);
    } else if (ad instanceof AccessorShortData sd) {
        // GL_UNSIGNED_SHORT -> mask to 0..65535
        for (int i = 0; i < n; i++) ib.put(sd.get(i, 0) & 0xFFFF);
    } else if (ad instanceof AccessorIntData id) {
        // GL_UNSIGNED_INT / INT -> just read
        for (int i = 0; i < n; i++) ib.put(id.get(i, 0));
    } else {
        throw new IllegalStateException("Unsupported index accessor type: " + ad.getClass().getName());
    }

    ib.rewind();
    return ib;
}

}

    // ------------------------------- Tri–box SAT -------------------------------

    private static boolean overlapsTriangle(int vx, int vy, int vz, Tri tri) {
        Vec3 c = new Vec3(vx + 0.5f, vy + 0.5f, vz + 0.5f);
        float r = 0.5f;
        Vec3 v0 = tri.v0.sub(c), v1 = tri.v1.sub(c), v2 = tri.v2.sub(c);
        Vec3 e0 = v1.sub(v0), e1 = v2.sub(v1), e2 = v0.sub(v2);

        // 1) plane normal test
        Vec3 n = e0.cross(e1);
        if (!axisPlaneOverlap(n, v0, r)) return false;

        // 2) 9 edge cross tests
        if (!edgeAxisTest(e0, v0, v1, v2, r)) return false;
        if (!edgeAxisTest(e1, v0, v1, v2, r)) return false;
        if (!edgeAxisTest(e2, v0, v1, v2, r)) return false;

        // 3) box axes
        if (!axisSlabOverlap(new Vec3(1,0,0), v0, v1, v2, r)) return false;
        if (!axisSlabOverlap(new Vec3(0,1,0), v0, v1, v2, r)) return false;
        if (!axisSlabOverlap(new Vec3(0,0,1), v0, v1, v2, r)) return false;

        return true;
    }

    private static boolean axisPlaneOverlap(Vec3 n, Vec3 v0, float r) {
        float d = Math.abs(dot(n, v0));
        float rProj = r * (Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z));
        return d <= rProj + 1e-6f;
    }

    private static boolean edgeAxisTest(Vec3 edge, Vec3 v0, Vec3 v1, Vec3 v2, float r) {
        Vec3[] axes = { new Vec3(1,0,0), new Vec3(0,1,0), new Vec3(0,0,1) };
        for (Vec3 a : axes) {
            Vec3 axis = edge.cross(a);
            float l2 = axis.x*axis.x + axis.y*axis.y + axis.z*axis.z;
            if (l2 < 1e-12f) continue;
            float p0 = dot(axis, v0), p1 = dot(axis, v1), p2 = dot(axis, v2);
            float min = Math.min(p0, Math.min(p1, p2));
            float max = Math.max(p0, Math.max(p1, p2));
            float rProj = r * (Math.abs(axis.x) + Math.abs(axis.y) + Math.abs(axis.z));
            if (min > rProj || max < -rProj) return false;
        }
        return true;
    }

    private static boolean axisSlabOverlap(Vec3 axis, Vec3 v0, Vec3 v1, Vec3 v2, float r) {
        float p0 = dot(axis, v0), p1 = dot(axis, v1), p2 = dot(axis, v2);
        float min = Math.min(p0, Math.min(p1, p2));
        float max = Math.max(p0, Math.max(p1, p2));
        return !(min > r || max < -r);
    }

    private static float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
    private static String baseName(File f) {
        String s = f.getName();
        int dot = s.lastIndexOf('.');
        return (dot >= 0) ? s.substring(0, dot) : s;
    }
}
