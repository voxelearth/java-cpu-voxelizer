# java-cpu-voxelizer

Tiny CLI to voxelize a GLB mesh on CPU and emit the same files your CUDA tool produces:

* `name_128.json` — `{ "blocks": { "1":[255,255,255] }, "xyzi":[[x,y,z,1], ...] }`
* `name_position.json` — `[ { "translation":[ox,oy,oz], "origin":[0,0,0] } ]`

Uses a solid triangle voxelizer with the hard-coded 3D Tiles cube (46×38×24 scaled by 1.3).

---

## Requirements

* JDK 17+
* Maven 3.8+

*(On Windows/WSL: build in the project directory; running from WSL is fine.)*

## Build

```bash
mvn -q -DskipTests clean package
```

Build artifact: `target/voxelizer-cli-1.0.0-all.jar`

## Usage

```bash
# basic
java -jar target/voxelizer-cli-1.0.0-all.jar \
  -f /path/to/tile.glb -s 128 -o out -3dtiles -v

# multiple files
java -jar target/voxelizer-cli-1.0.0-all.jar \
  -f a.glb -f b.glb -f c.glb -s 128 -o out -3dtiles

# from a list (one GLB path per line)
java -jar target/voxelizer-cli-1.0.0-all.jar \
  -filelist tiles.txt -s 128 -o out --no-3dtiles
```

### Options

* `-f <path>`  Add a GLB file (repeatable)
* `-filelist <txt>` Text file with one GLB path per line
* `-s <grid>`  Grid size (default `128`)
* `-o <dir>`  Output directory (default `./out`)
* `-3dtiles` / `--no-3dtiles` Enable/disable 3D Tiles cube bounds (default **on**)
* `-v`  Verbose logs
* `-h` Help

## Output

For each input `name.glb`, you’ll get in `<out>`:

* `name_<grid>.json`
* `name_position.json`

## Quick compare vs CUDA output

```bash
# optional: compare only geometry (ignores color palette)
jq '.xyzi | sort_by(.[0],.[1],.[2])' out_cpu/name_128.json  > cpu.xyzi.json
jq '.xyzi | sort_by(.[0],.[1],.[2])' out_cuda/name_128.json > gpu.xyzi.json
diff -u gpu.xyzi.json cpu.xyzi.json
```

That’s it!
