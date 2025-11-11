package com.example.voxelizer;

import de.javagl.jgltf.model.GltfModel;
import de.javagl.jgltf.model.SceneModel;
import de.javagl.jgltf.model.NodeModel;
import de.javagl.jgltf.model.MeshModel;
import de.javagl.jgltf.model.MeshPrimitiveModel;
import de.javagl.jgltf.model.AccessorModel;
import de.javagl.jgltf.model.AccessorData;
import de.javagl.jgltf.model.AccessorDatas;
import de.javagl.jgltf.model.AccessorFloatData;
import de.javagl.jgltf.model.AccessorByteData;
import de.javagl.jgltf.model.AccessorShortData;
import de.javagl.jgltf.model.AccessorIntData;
import de.javagl.jgltf.model.ImageModel;
import de.javagl.jgltf.model.io.GltfModelReader;

import org.json.JSONArray;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class JavaCpuVoxelizer {

    private final int grid;
    private final boolean tiles3d;
    private final boolean verbose;

    // Hardcoded 3D Tiles cube (match CUDA)
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

    // ------------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------------
    public Stats voxelizeSingleGLB(File glbFile, File outDir) throws Exception {
        if (verbose) System.out.println("[Load] " + glbFile.getAbsolutePath());

        // Load model and decode a single global texture (1st image), like CUDA path
        LoaderResult lr = GltfReader.loadTrianglesAndGlobalTexture(glbFile);

        if (lr.mesh.tris.isEmpty()) throw new IllegalStateException("No triangles in " + glbFile);

        // BBox (file coordinates, no node transforms), then 3D Tiles rect → cube
        Aabb baseBox = lr.mesh.computeBounds();
        Aabb voxelRect = tiles3d ? makeTilesBoxAroundMeshCenter(baseBox) : baseBox;
        Aabb bbox = makeCube(voxelRect);

        if (verbose) {
            System.out.printf("[BBox] tiles=%s cube min=(%.6f,%.6f,%.6f)  max=(%.6f,%.6f,%.6f)%n",
                    tiles3d, bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
        }

        // Unit from cube side / grid (same formula as your original)
        Vec3 size = bbox.extent();
        float unit = Math.max(Math.max(size.x, size.y), size.z) / grid;
        if (verbose) System.out.printf("[Grid] %d^3  unit=%.6f%n", grid, unit);

        // Translation EXACTLY as original
        int ox = Math.round((-bbox.min.x) / unit);
        int oy = Math.round((-bbox.min.y) / unit);
        int oz = Math.round((-bbox.min.z) / unit);

        // Voxel-space transform of verts
        for (Tri t : lr.mesh.tris) {
            t.v0 = t.v0.sub(bbox.min).div(unit);
            t.v1 = t.v1.sub(bbox.min).div(unit);
            t.v2 = t.v2.sub(bbox.min).div(unit);
            t.tex = lr.globalTexture; // same texture for all tris (CUDA-like)
        }

        final int total = grid * grid * grid;
        BitSet occ = new BitSet(total);
        int[] colorBuf = new int[total]; // 0 = unset; else 0xRRGGBB

        long t0 = System.nanoTime();
        int triCount = lr.mesh.tris.size();

        final int PAD = 1; // conservative bounds
        for (Tri t : lr.mesh.tris) {
            float minx = Math.min(t.v0.x, Math.min(t.v1.x, t.v2.x));
            float miny = Math.min(t.v0.y, Math.min(t.v1.y, t.v2.y));
            float minz = Math.min(t.v0.z, Math.min(t.v1.z, t.v2.z));
            float maxx = Math.max(t.v0.x, Math.max(t.v1.x, t.v2.x));
            float maxy = Math.max(t.v0.y, Math.max(t.v1.y, t.v2.y));
            float maxz = Math.max(t.v0.z, Math.max(t.v1.z, t.v2.z));

            int x0 = clamp((int)Math.floor(minx) - PAD, 0, grid - 1);
            int y0 = clamp((int)Math.floor(miny) - PAD, 0, grid - 1);
            int z0 = clamp((int)Math.floor(minz) - PAD, 0, grid - 1);
            int x1 = clamp((int)Math.ceil (maxx) + PAD, 0, grid - 1);
            int y1 = clamp((int)Math.ceil (maxy) + PAD, 0, grid - 1);
            int z1 = clamp((int)Math.ceil (maxz) + PAD, 0, grid - 1);

            for (int z = z0; z <= z1; z++) {
                for (int y = y0; y <= y1; y++) {
                    for (int x = x0; x <= x1; x++) {
                        if (!overlapsTriangle(x, y, z, t)) continue;

                        int rgb = 0xFFFFFF; // default white if no tex/uv
                        if (t.hasUV && t.tex != null) {
                            Vec3 p = new Vec3(x + 0.5f, y + 0.5f, z + 0.5f);
                            // barycentric in 3D voxel space
                            float[] vw = barycentricVW(t.v0, t.v1, t.v2, p);
                            float l1 = vw[0], l2 = vw[1], l0 = 1.0f - l1 - l2; // λ0,λ1,λ2

                            float u = l0 * t.tu0 + l1 * t.tu1 + l2 * t.tu2;
                            float v = l0 * t.tv0 + l1 * t.tv1 + l2 * t.tv2;

                            // clamp to [0,1] (CUDA clamps)
                            if (u < 0f) u = 0f; else if (u > 1f) u = 1f;
                            if (v < 0f) v = 0f; else if (v > 1f) v = 1f;

                            rgb = bilinearSampleRGB(t.tex, u, v); // V flip inside
                        }

                        int lin = x + grid * (y + grid * z);
                        occ.set(lin);
                        colorBuf[lin] = rgb; // last-wins, like GPU kernel
                    }
                }
            }
        }

        long t1 = System.nanoTime();
        int filled = occ.cardinality();
        if (verbose) System.out.printf("[Raster] tris=%d  filled=%d  time=%.1f ms%n",
                triCount, filled, (t1 - t0)/1e6);

        // Build palette & write JSON blocks/xyzi (CUDA-style)
        Map<Integer, Integer> colorToIndex = new LinkedHashMap<>();
        int nextIndex = 1;

        String base = baseName(glbFile);
        File jsonOut = new File(outDir, base + "_" + grid + ".json");
        JSONObject blocks = new JSONObject();
        JSONArray xyzi = new JSONArray();

        for (int z = 0; z < grid; z++) {
            for (int y = 0; y < grid; y++) {
                for (int x = 0; x < grid; x++) {
                    int lin = x + grid * (y + grid * z);
                    if (!occ.get(lin)) continue;

                    int rgb = colorBuf[lin] == 0 ? 0xFFFFFF : colorBuf[lin];

                    Integer idx = colorToIndex.get(rgb);
                    if (idx == null) {
                        idx = nextIndex++;
                        colorToIndex.put(rgb, idx);
                    }
                    xyzi.put(new JSONArray(new int[]{ x - ox, y - oy, z - oz, idx }));
                }
            }
        }
        for (Map.Entry<Integer,Integer> e : colorToIndex.entrySet()) {
            int rgb = e.getKey();
            int r = (rgb >> 16) & 255;
            int g = (rgb >> 8)  & 255;
            int b = (rgb)       & 255;
            blocks.put(Integer.toString(e.getValue()), new JSONArray(new int[]{r,g,b}));
        }

        JSONObject root = new JSONObject();
        root.put("blocks", blocks);
        root.put("xyzi", xyzi);
        try (FileWriter fw = new FileWriter(jsonOut)) { fw.write(root.toString()); }
        if (verbose) System.out.println("[Write] " + jsonOut.getName() +
                " (voxels=" + xyzi.length() + ", colors=" + colorToIndex.size() + ")");

        // position file
        File posOut = new File(outDir, base + "_position.json");
        JSONArray arr = new JSONArray();
        JSONObject pos = new JSONObject();
        pos.put("translation", new JSONArray(new int[]{ox, oy, oz}));
        pos.put("origin", new JSONArray(new int[]{0, 0, 0}));
        arr.put(pos);
        try (FileWriter fw = new FileWriter(posOut)) { fw.write(arr.toString()); }
        if (verbose) System.out.println("[Write] " + posOut.getName() +
                "  translation=["+ox+","+oy+","+oz+"] origin=[0,0,0]");

        return new Stats(base, grid, triCount, filled, ox, oy, oz);
    }

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------
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
        Vec3 v0, v1, v2;         // voxel-space verts (after bbox/unit)
        boolean hasUV;
        float tu0, tv0, tu1, tv1, tu2, tv2;
        BufferedImage tex;       // global texture (first image)
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
    private static final class LoaderResult {
        final Mesh mesh; final BufferedImage globalTexture;
        LoaderResult(Mesh m, BufferedImage t){ this.mesh=m; this.globalTexture=t; }
    }

    private static Aabb makeCube(Aabb box) {
        Vec3 c = box.center();
        Vec3 ext = box.extent();
        float side = Math.max(ext.x, Math.max(ext.y, ext.z));
        Vec3 half = new Vec3(side/2f, side/2f, side/2f);
        return new Aabb(c.sub(half), c.add(half));
    }
    private static Aabb makeTilesBoxAroundMeshCenter(Aabb meshBox) {
        Vec3 c = meshBox.center();
        Vec3 half = new Vec3(TILE_X/2f, TILE_Y/2f, TILE_Z/2f);
        return new Aabb(c.sub(half), c.add(half));
    }

    // ------------------------------------------------------------------------
    // GLB reader (NO node transforms) + FIRST IMAGE texture (CUDA-like)
    // ------------------------------------------------------------------------
    private static final class GltfReader {
        static LoaderResult loadTrianglesAndGlobalTexture(File glbFile) throws Exception {
            byte[] data = Files.readAllBytes(glbFile.toPath());
            GltfModel model = new GltfModelReader().readWithoutReferences(new ByteArrayInputStream(data));

            Mesh out = new Mesh();

            // decode first image in the glTF as global texture
            BufferedImage globalTex = decodeFirstImage(model);

            List<SceneModel> scenes = model.getSceneModels();
            if (scenes == null || scenes.isEmpty()) {
                for (NodeModel root : model.getNodeModels()) {
                    traverseNode(root, out, globalTex);
                }
            } else {
                for (SceneModel scene : scenes) {
                    for (NodeModel root : scene.getNodeModels()) {
                        traverseNode(root, out, globalTex);
                    }
                }
            }
            return new LoaderResult(out, globalTex);
        }

        private static void traverseNode(NodeModel node, Mesh out, BufferedImage globalTex) {
            List<MeshModel> meshes = node.getMeshModels();
            if (meshes != null) {
                for (MeshModel meshModel : meshes) {
                    for (MeshPrimitiveModel prim : meshModel.getMeshPrimitiveModels()) {
                        addPrimitive(prim, out, globalTex);
                    }
                }
            }
            for (NodeModel child : node.getChildren()) {
                traverseNode(child, out, globalTex);
            }
        }

        private static void addPrimitive(MeshPrimitiveModel pm, Mesh out, BufferedImage globalTex) {
            AccessorModel posAcc = pm.getAttributes().get("POSITION");
            if (posAcc == null) return;

            AccessorFloatData pos = AccessorDatas.createFloat(posAcc);

            AccessorModel uvAcc = pm.getAttributes().get("TEXCOORD_0");
            AccessorFloatData uvs = (uvAcc != null) ? AccessorDatas.createFloat(uvAcc) : null;

            IntBuffer indices = toIndexBuffer(pm);

            if (indices != null) {
                while (indices.hasRemaining()) {
                    int i0 = indices.get(), i1 = indices.get(), i2 = indices.get();
                    Tri t = new Tri();
                    t.v0 = new Vec3(pos.get(i0,0), pos.get(i0,1), pos.get(i0,2));
                    t.v1 = new Vec3(pos.get(i1,0), pos.get(i1,1), pos.get(i1,2));
                    t.v2 = new Vec3(pos.get(i2,0), pos.get(i2,1), pos.get(i2,2));
                    if (uvs != null) {
                        t.hasUV = true;
                        t.tu0 = uvs.get(i0,0); t.tv0 = uvs.get(i0,1);
                        t.tu1 = uvs.get(i1,0); t.tv1 = uvs.get(i1,1);
                        t.tu2 = uvs.get(i2,0); t.tv2 = uvs.get(i2,1);
                        t.tex = globalTex;
                    }
                    out.tris.add(t);
                }
            } else {
                int count = pos.getNumElements();
                for (int i = 0; i + 2 < count; i += 3) {
                    Tri t = new Tri();
                    t.v0 = new Vec3(pos.get(i,0),   pos.get(i,1),   pos.get(i,2));
                    t.v1 = new Vec3(pos.get(i+1,0), pos.get(i+1,1), pos.get(i+1,2));
                    t.v2 = new Vec3(pos.get(i+2,0), pos.get(i+2,1), pos.get(i+2,2));
                    if (uvs != null) {
                        t.hasUV = true;
                        t.tu0 = uvs.get(i,0);   t.tv0 = uvs.get(i,1);
                        t.tu1 = uvs.get(i+1,0); t.tv1 = uvs.get(i+1,1);
                        t.tu2 = uvs.get(i+2,0); t.tv2 = uvs.get(i+2,1);
                        t.tex = globalTex;
                    }
                    out.tris.add(t);
                }
            }
        }

        private static BufferedImage decodeFirstImage(GltfModel model) throws Exception {
            List<ImageModel> imgs = model.getImageModels();
            if (imgs == null || imgs.isEmpty()) return null;
            for (ImageModel im : imgs) {
                ByteBuffer bb = im.getImageData();
                if (bb == null) continue;
                byte[] arr = new byte[bb.remaining()];
                bb.get(arr);
                try (ByteArrayInputStream bais = new ByteArrayInputStream(arr)) {
                    BufferedImage img = ImageIO.read(bais);
                    if (img != null) return img; // first decodable image
                }
            }
            return null;
        }

        private static IntBuffer toIndexBuffer(MeshPrimitiveModel pm) {
            AccessorModel idxAcc = pm.getIndices();
            if (idxAcc == null) return null;

            AccessorData ad = AccessorDatas.create(idxAcc);
            int n = ad.getNumElements();
            IntBuffer ib = IntBuffer.allocate(n);

            if (ad instanceof AccessorByteData bd) {
                for (int i = 0; i < n; i++) ib.put(bd.get(i, 0) & 0xFF);
            } else if (ad instanceof AccessorShortData sd) {
                for (int i = 0; i < n; i++) ib.put(sd.get(i, 0) & 0xFFFF);
            } else if (ad instanceof AccessorIntData id) {
                for (int i = 0; i < n; i++) ib.put(id.get(i, 0));
            } else {
                throw new IllegalStateException("Unsupported index accessor type: " + ad.getClass().getName());
            }

            ib.rewind();
            return ib;
        }
    }

    // ---- math/sampling ----
    /** Return [λ1, λ2]; λ0 = 1 - λ1 - λ2. */
    private static float[] barycentricVW(Vec3 a, Vec3 b, Vec3 c, Vec3 p) {
        float v0x=b.x-a.x, v0y=b.y-a.y, v0z=b.z-a.z;
        float v1x=c.x-a.x, v1y=c.y-a.y, v1z=c.z-a.z;
        float v2x=p.x-a.x, v2y=p.y-a.y, v2z=p.z-a.z;

        float d00 = v0x*v0x + v0y*v0y + v0z*v0z;
        float d01 = v0x*v1x + v0y*v1y + v0z*v1z;
        float d11 = v1x*v1x + v1y*v1y + v1z*v1z;
        float d20 = v2x*v0x + v2y*v0y + v2z*v0z;
        float d21 = v2x*v1x + v2y*v1y + v2z*v1z;
        float denom = d00 * d11 - d01 * d01;
        if (Math.abs(denom) < 1e-20f) return new float[]{0.33f,0.33f};
        float v = (d11 * d20 - d01 * d21) / denom; // λ1
        float w = (d00 * d21 - d01 * d20) / denom; // λ2
        return new float[]{v, w};
    }

    /** Bilinear sample; flips V like CUDA. Returns 0xRRGGBB. */
    private static int bilinearSampleRGB(BufferedImage img, float u, float v01) {
        int w = img.getWidth(), h = img.getHeight();
        if (w <= 0 || h <= 0) return 0xFFFFFF;

        float x = u * (w - 1);
        // float y = (1.0f - v01) * (h - 1); // V flip
        float y = v01 * (h - 1); // no V flip

        int x0 = (int)Math.floor(x);
        int y0 = (int)Math.floor(y);
        int x1 = Math.min(x0 + 1, w - 1);
        int y1 = Math.min(y0 + 1, h - 1);
        float dx = x - x0;
        float dy = y - y0;

        int c00 = img.getRGB(clamp(x0,0,w-1), clamp(y0,0,h-1));
        int c10 = img.getRGB(clamp(x1,0,w-1), clamp(y0,0,h-1));
        int c01 = img.getRGB(clamp(x0,0,w-1), clamp(y1,0,h-1));
        int c11 = img.getRGB(clamp(x1,0,w-1), clamp(y1,0,h-1));

        float r00 = (c00 >> 16 & 255), g00 = (c00 >> 8 & 255), b00 = (c00 & 255);
        float r10 = (c10 >> 16 & 255), g10 = (c10 >> 8 & 255), b10 = (c10 & 255);
        float r01 = (c01 >> 16 & 255), g01 = (c01 >> 8 & 255), b01 = (c01 & 255);
        float r11 = (c11 >> 16 & 255), g11 = (c11 >> 8 & 255), b11 = (c11 & 255);

        float r0 = r00*(1-dx) + r10*dx;
        float g0 = g00*(1-dx) + g10*dx;
        float b0 = b00*(1-dx) + b10*dx;

        float r1 = r01*(1-dx) + r11*dx;
        float g1 = g01*(1-dx) + g11*dx;
        float b1 = b01*(1-dx) + b11*dx;

        int r = (int)Math.round(r0*(1-dy) + r1*dy);
        int g = (int)Math.round(g0*(1-dy) + g1*dy);
        int b = (int)Math.round(b0*(1-dy) + b1*dy);

        if (r<0) r=0; else if (r>255) r=255;
        if (g<0) g=0; else if (g>255) g=255;
        if (b<0) b=0; else if (b>255) b=255;
        return (r<<16)|(g<<8)|b;
    }

    // ------------------------------------------------------------------------
    // SAT tests
    // ------------------------------------------------------------------------
    private static boolean overlapsTriangle(int vx, int vy, int vz, Tri tri) {
        Vec3 c = new Vec3(vx + 0.5f, vy + 0.5f, vz + 0.5f);
        float r = 0.5f;
        Vec3 v0 = tri.v0.sub(c), v1 = tri.v1.sub(c), v2 = tri.v2.sub(c);
        Vec3 e0 = v1.sub(v0), e1 = v2.sub(v1), e2 = v0.sub(v2);

        Vec3 n = e0.cross(e1);
        if (!axisPlaneOverlap(n, v0, r)) return false;

        if (!edgeAxisTest(e0, v0, v1, v2, r)) return false;
        if (!edgeAxisTest(e1, v0, v1, v2, r)) return false;
        if (!edgeAxisTest(e2, v0, v1, v2, r)) return false;

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
