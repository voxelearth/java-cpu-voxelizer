package com.example.voxelizer;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class Main {

    private static void help() {
        System.out.println("""
            Usage:
              java -jar voxelizer-cli-1.0.0-all.jar -f <file.glb> [-f <file2.glb> ...]
                                                    [-filelist <list.txt>]
                                                    [-s <grid>] [-o <outdir>]
                                                    [-3dtiles] [--no-3dtiles]
                                                    [- j | --jobs <n>] [-v] [-h]

            Options:
              -f <path>            GLB file (repeatable)
              -filelist <txt>      Text file with one GLB path per line
              -s <grid>            Grid size (default 128)
              -o <dir>             Output directory (default ./out)
              -3dtiles             Enable 3D Tiles cube bbox (default ON)
              --no-3dtiles         Disable 3D Tiles bbox
              -j, --jobs <n>       Parallel jobs (default = #CPUs, min 1)
              -v                   Verbose logging
              -h                   Show this help
            """);
    }

    public static void main(String[] args) {
        if (args.length == 0) { help(); System.exit(1); }

        List<File> inputs = new ArrayList<>();
        File outDir = new File("out");
        int grid = 128;
        boolean tiles3d = true;
        boolean verbose = false;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors());

        try {
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                switch (a) {
                    case "-h" -> { help(); return; }
                    case "-v" -> verbose = true;
                    case "-3dtiles" -> tiles3d = true;
                    case "--no-3dtiles" -> tiles3d = false;
                    case "-s" -> {
                        if (i + 1 >= args.length) throw new IllegalArgumentException("Missing value for -s");
                        grid = Integer.parseInt(args[++i]);
                    }
                    case "-o" -> {
                        if (i + 1 >= args.length) throw new IllegalArgumentException("Missing value for -o");
                        outDir = new File(args[++i]);
                    }
                    case "-j", "--jobs" -> {
                        if (i + 1 >= args.length) throw new IllegalArgumentException("Missing value for -j/--jobs");
                        jobs = Math.max(1, Integer.parseInt(args[++i]));
                    }
                    case "-f" -> {
                        if (i + 1 >= args.length) throw new IllegalArgumentException("Missing value for -f");
                        inputs.add(new File(args[++i]));
                    }
                    case "-filelist" -> {
                        if (i + 1 >= args.length) throw new IllegalArgumentException("Missing value for -filelist");
                        File list = new File(args[++i]);
                        for (String line : Files.readAllLines(list.toPath())) {
                            String p = line.trim();
                            if (!p.isEmpty()) inputs.add(new File(p));
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
            System.out.println("[CLI] Jobs=" + jobs);
            System.out.println("[CLI] Output: " + outDir.getAbsolutePath());

            // Make captured vars effectively final for lambdas
            final int gridF = grid;
            final boolean tiles3dF = tiles3d;
            final boolean verboseF = verbose;
            final File outDirF = outDir;
            final int jobsF = jobs;

            long t0All = System.nanoTime();

            ExecutorService pool = (jobsF > 1)
                    ? Executors.newWorkStealingPool(jobsF)
                    : Executors.newSingleThreadExecutor();

            List<Future<JavaCpuVoxelizer.Stats>> futures = new ArrayList<>(inputs.size());
            for (File f : inputs) {
                final File file = f; // capture per-iteration variable for lambda
                futures.add(pool.submit(() -> {
                    long t0 = System.nanoTime();
                    JavaCpuVoxelizer vox = new JavaCpuVoxelizer(gridF, tiles3dF, /*verbose*/ verboseF);
                    JavaCpuVoxelizer.Stats stats = vox.voxelizeSingleGLB(file, outDirF);
                    long t1 = System.nanoTime();
                    System.out.printf("[OK] %-40s  tris=%d  filled=%d  time=%.1f ms%n",
                            file.getName(), stats.triangles, stats.filled, (t1 - t0) / 1e6);
                    System.out.printf("     → %s_%d.json%n", stats.baseName, gridF);
                    return stats;
                }));
            }

            int totalFilled = 0;
            int done = 0;
            for (Future<JavaCpuVoxelizer.Stats> fut : futures) {
                try {
                    JavaCpuVoxelizer.Stats s = fut.get();
                    totalFilled += s.filled;
                    done++;
                } catch (ExecutionException ee) {
                    System.err.println("[ERR] " + ee.getCause());
                    ee.getCause().printStackTrace();
                }
            }
            pool.shutdown();

            long t1All = System.nanoTime();
            System.out.printf("[ALL DONE] files=%d/%d filled_total=%d time=%.1f ms%n",
                    done, inputs.size(), totalFilled, (t1All - t0All) / 1e6);

        } catch (Throwable e) {
            System.err.println("[ERR] " + e.getMessage());
            e.printStackTrace();
            System.exit(10);
        }
    }
}
