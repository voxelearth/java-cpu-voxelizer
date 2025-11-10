package com.example.voxelizer;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static void help() {
        System.out.println("""
            Usage:
              java -jar voxelizer-cli-1.0.0-all.jar -f <file.glb> [-f <file2.glb> ...]
                                               [-filelist <list.txt>]
                                               [-s 128] [-o outdir]
                                               [-3dtiles] [--no-3dtiles]
                                               [-v] [-h]

            Options:
              -f <path>            GLB file (repeatable)
              -filelist <txt>      Text file with one GLB path per line
              -s <grid>            Grid size (default 128)
              -o <dir>             Output directory (default ./out)
              -3dtiles             Enforce 3D Tiles cube bounds (default ON)
              --no-3dtiles         Disable 3D Tiles cube bounds
              -v                   Verbose debug logging
              -h                   Show help
            """);
    }

    public static void main(String[] args) {
        if (args.length == 0) { help(); System.exit(1); }

        List<File> inputs = new ArrayList<>();
        File outDir = new File("out");
        int grid = 128;
        boolean tiles3d = true;
        boolean verbose = false;

        try {
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                switch (a) {
                    case "-h" -> { help(); return; }
                    case "-v" -> verbose = true;
                    case "-3dtiles" -> tiles3d = true;
                    case "--no-3dtiles" -> tiles3d = false;
                    case "-s" -> grid = Integer.parseInt(args[++i]);
                    case "-o" -> outDir = new File(args[++i]);
                    case "-f" -> inputs.add(new File(args[++i]));
                    case "-filelist" -> {
                        File list = new File(args[++i]);
                        for (String line : Files.readAllLines(list.toPath())) {
                            line = line.trim();
                            if (!line.isEmpty()) inputs.add(new File(line));
                        }
                    }
                    default -> {
                        if (a.startsWith("-")) {
                            System.err.println("Unknown option: " + a);
                            help();
                            System.exit(2);
                        } else {
                            // bare file path convenience
                            inputs.add(new File(a));
                        }
                    }
                }
            }

            if (inputs.isEmpty()) {
                System.err.println("No input files.");
                help();
                System.exit(2);
            }
            if (!outDir.exists() && !outDir.mkdirs()) {
                System.err.println("Cannot create output dir: " + outDir.getAbsolutePath());
                System.exit(3);
            }

            System.out.println("[CLI] Grid=" + grid + ", 3DTiles=" + tiles3d + ", Verbose=" + verbose);
            System.out.println("[CLI] Output: " + outDir.getAbsolutePath());

            JavaCpuVoxelizer vox = new JavaCpuVoxelizer(grid, tiles3d, verbose);

            long t0All = System.nanoTime();
            int totalFilled = 0;
            for (File f : inputs) {
                long t0 = System.nanoTime();
                var stats = vox.voxelizeSingleGLB(f, outDir);
                long t1 = System.nanoTime();
                totalFilled += stats.filled;
                System.out.printf("[OK] %-40s  tris=%d  filled=%d  time=%.1f ms%n",
                        f.getName(), stats.triangles, stats.filled, (t1 - t0) / 1e6);
                System.out.printf("     → %s_%d.json, %s_position.json%n",
                        stats.baseName, grid, stats.baseName);
            }
            long t1All = System.nanoTime();
            System.out.printf("[ALL DONE] files=%d filled_total=%d time=%.1f ms%n",
                    inputs.size(), totalFilled, (t1All - t0All) / 1e6);

        } catch (Exception e) {
            System.err.println("[ERR] " + e.getMessage());
            e.printStackTrace();
            System.exit(10);
        }
    }
}
