#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
third_party_root="$repo_root/extra_resources"
flashsac_root="$third_party_root/FlashSAC"
genesis_root="$third_party_root/Genesis"

mkdir -p "$third_party_root"

if [[ ! -d "$flashsac_root/.git" ]]; then
  git clone https://github.com/Holiday-Robot/FlashSAC.git "$flashsac_root"
fi
git -C "$flashsac_root" checkout 87edc9061150ae9e962dd84e6544e27a1554b3ab
git -C "$flashsac_root" apply "$repo_root/third_party/flashsac.patch"

if [[ ! -d "$genesis_root/.git" ]]; then
  git clone https://github.com/Genesis-Embodied-AI/Genesis.git "$genesis_root"
fi
git -C "$genesis_root" checkout 6b75c7bdc37147627e6e718765c83ee90be696e3

printf 'Third-party sources are ready under %s\n' "$third_party_root"

