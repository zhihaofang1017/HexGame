base_dir="$1"
if [ -z "$base_dir" ]; then
    base_dir="$HOME"
fi
rm -rf "$base_dir/HexGame"
mkdir -p "$base_dir/HexGame"
cp -R agents src Hex.py run.sh "$base_dir/HexGame/"
