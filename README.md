# Melee

## Prerequisites

- **Dolphin Emulator** (NoGUI or Headless build)
- **Super Smash Bros. Melee ISO** (`ssbm.iso`)
- **Python 3.10** environment with the `libmelee` package installed (Already in the repo)
- **Slippi** integration enabled in Dolphin

## File Placement

1. **ISO File**  
   Place `ssbm.iso` in the project’s root directory (next to your Python scripts).

2. **Dolphin Executable and Supporting Files**  
   - Store the Dolphin executable (and any needed libraries) under `Images/<MODE>/squashfs-root/usr/bin/`.  
   - Replace `<MODE>` with either `NoGui` (for a windowed emulator) or `Headless` (for a headless build).  
   - Example path for NoGUI mode:  
     ```
     Images/NoGui/squashfs-root/usr/bin/dolphin-emu
     ```

## Configuration

All runtime options live in a single Python dictionary called `CONFIG`. You can modify the following fields as needed:

- **dolphin_path**:  
  Full path to the Dolphin executable.  
  ```python
  "dolphin_path": "Images/NoGui/squashfs-root/usr/bin/dolphin-emu"
