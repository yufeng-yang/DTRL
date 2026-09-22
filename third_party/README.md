# Third-party setup

The Unitree Go2 experiments build on
[FlashSAC](https://github.com/Holiday-Robot/FlashSAC) and
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis). Their full source
trees and local virtual environments are not vendored in this repository.

The experiments used:

- FlashSAC commit `87edc9061150ae9e962dd84e6544e27a1554b3ab`
- Genesis commit `6b75c7bdc37147627e6e718765c83ee90be696e3`
- the DTRL-specific FlashSAC changes in `flashsac.patch`

From the repository root, prepare the source trees with:

```bash
bash scripts/setup_third_party.sh
```

The script clones the exact revisions into `extra_resources/` and applies the
DTRL patch to FlashSAC. Review the upstream projects' installation
instructions for system and GPU prerequisites.

