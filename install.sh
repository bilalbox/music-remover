#!/usr/bin/env bash
# Builds music-remover and symlinks it into ~/.local/bin so it's runnable from anywhere.
# Fetches and exports the HTDemucs-DnR model first if it isn't present yet.
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
bindir="$HOME/.local/bin"

if [ ! -f "$root/models/htdemucs_dnr/htdemucs_fwd.xml" ]; then
    "$root/fetch-model.sh"
fi

cargo build --release --manifest-path "$root/Cargo.toml"

mkdir -p "$bindir"
ln -sf "$root/target/release/music-remover" "$bindir/music-remover"

echo ">> linked $bindir/music-remover -> $root/target/release/music-remover"
case ":$PATH:" in
    *":$bindir:"*) echo ">> music-remover is ready: music-remover --device CPU <input>" ;;
    *) echo ">> $bindir is not on your PATH — add it to your shell rc, e.g.:"
       echo "   export PATH=\"$bindir:\$PATH\"" ;;
esac
