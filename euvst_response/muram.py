"""
Turn a MURaM snapshot into an ECLIPSE atmosphere file, and describe such files.

    eclipse-atmosphere from-muram --data-dir data/atmosphere --snapshot 0270000
    eclipse-atmosphere info data/atmosphere_0270000.h5

MURaM writes one Fortran-ordered binary per variable per snapshot with no
header of its own, so the cube shape and cell sizes have to be given. The
defaults are those of the Cheung et al. (2019) flare simulation ECLIPSE's
examples use. The file that comes out places the box as ECLIPSE always has:
x and y centred on zero, and z = 0 at the centre of the bottom cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import astropy.units as u
import numpy as np

from .atmosphere import AXES, Atmosphere, describe_atmosphere_file, write_atmosphere
from .synthesis import load_cube, read_timestep_time

# MURaM's files for each quantity, relative to the data directory and
# without the snapshot suffix, and the cgs units MURaM writes them in.
MURAM_FILES = {
    "temperature": "temp/eosT",
    "mass_density": "rho/result_prim_0",
    "velocity_x": "vx/result_prim_1",
    "velocity_y": "vy/result_prim_3",
    "velocity_z": "vz/result_prim_2",
}
MURAM_UNITS = {
    "temperature": u.K,
    "mass_density": u.g / u.cm**3,
    "velocity_x": u.cm / u.s,
    "velocity_y": u.cm / u.s,
    "velocity_z": u.cm / u.s,
}
MURAM_HEADER = "header/Header"

# The Cheung et al. (2019) run: 512 x 768 x 256 cells in the file's own
# (nx, nz, ny) order, 192 km across and 64 km up.
DEFAULT_SHAPE = (512, 768, 256)
DEFAULT_VOXEL = {"x": 0.192 * u.Mm, "y": 0.192 * u.Mm, "z": 0.064 * u.Mm}


def read_muram(
    data_dir: str | Path,
    snapshot: str,
    shape: Sequence[int] = DEFAULT_SHAPE,
    voxel_dx: u.Quantity = DEFAULT_VOXEL["x"],
    voxel_dy: u.Quantity = DEFAULT_VOXEL["y"],
    voxel_dz: u.Quantity = DEFAULT_VOXEL["z"],
    files: Optional[dict] = None,
    velocities: Sequence[str] = AXES,
    header: Optional[str] = MURAM_HEADER,
    time: Optional[u.Quantity] = None,
    source: str = "MURaM",
) -> Atmosphere:
    """
    Read one MURaM snapshot as an :class:`~euvst_response.atmosphere.Atmosphere`.

    Parameters
    ----------
    data_dir : str or Path
        The directory the file names are relative to.
    snapshot : str
        The suffix naming the snapshot, such as ``"0270000"``.
    shape : sequence of int
        The cube dimensions in the file's own order, ``(nx, nz, ny)``.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        The cell sizes.
    files : dict, optional
        File name prefixes for any of the quantities in :data:`MURAM_FILES`,
        overriding the defaults there.
    velocities : sequence of str
        Which velocity components to read. A synthesis needs only the one
        along its line of sight, and each is 400 MB at full resolution.
    header : str, optional
        The prefix of the header file the snapshot time is read from, or
        None not to look for one. A missing file leaves the time unset.
    time : u.Quantity, optional
        The snapshot time, overriding the header.
    source : str
        What to record as the atmosphere's source.
    """
    data_dir = Path(data_dir)
    names = {**MURAM_FILES, **(files or {})}
    wanted = ["temperature", "mass_density"] + [f"velocity_{axis}" for axis in velocities]
    fields = {}
    for name in wanted:
        path = data_dir / f"{names[name]}.{snapshot}"
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        fields[name] = load_cube(path, shape=tuple(shape), unit=MURAM_UNITS[name])

    # Where ECLIPSE has always placed a MURaM box: x and y centred on zero,
    # and the centre of the bottom cell at z = 0.
    nz, ny, nx = fields["temperature"].shape
    x_edges = (np.arange(nx + 1) - nx / 2) * voxel_dx
    y_edges = (np.arange(ny + 1) - ny / 2) * voxel_dy
    z_edges = (np.arange(nz + 1) - 0.5) * voxel_dz

    if time is None and header is not None:
        header_path = data_dir / f"{header}.{snapshot}"
        if header_path.exists():
            time = read_timestep_time(header_path) * u.s

    return Atmosphere(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
                      time=time, source=source, **fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eclipse-atmosphere",
        description="Write and inspect ECLIPSE atmosphere files.")
    commands = parser.add_subparsers(dest="command", required=True)

    muram = commands.add_parser(
        "from-muram", help="Convert one MURaM snapshot to an atmosphere file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    muram.add_argument("--data-dir", type=str, default="data/atmosphere",
                       help="Directory containing the MURaM files")
    muram.add_argument("--snapshot", type=str, required=True,
                       help="Snapshot suffix of the files, e.g. 0270000")
    muram.add_argument("--output", type=str, default=None,
                       help="Atmosphere file to write (default: "
                            "<data-dir>_<snapshot>.h5 next to the data directory)")
    muram.add_argument("--cube-shape", nargs=3, type=int, default=list(DEFAULT_SHAPE),
                       help="Cube dimensions in the file's storage order (nx nz ny)")
    muram.add_argument("--voxel-dx", type=str, default=str(DEFAULT_VOXEL["x"]),
                       help="Cell size in x")
    muram.add_argument("--voxel-dy", type=str, default=str(DEFAULT_VOXEL["y"]),
                       help="Cell size in y")
    muram.add_argument("--voxel-dz", type=str, default=str(DEFAULT_VOXEL["z"]),
                       help="Cell size in z")
    muram.add_argument("--temp-file", type=str, default=MURAM_FILES["temperature"],
                       help="Temperature file relative to data-dir, without the suffix")
    muram.add_argument("--rho-file", type=str, default=MURAM_FILES["mass_density"],
                       help="Density file relative to data-dir, without the suffix")
    muram.add_argument("--vx-file", type=str, default=MURAM_FILES["velocity_x"],
                       help="Velocity x file relative to data-dir, without the suffix")
    muram.add_argument("--vy-file", type=str, default=MURAM_FILES["velocity_y"],
                       help="Velocity y file relative to data-dir, without the suffix")
    muram.add_argument("--vz-file", type=str, default=MURAM_FILES["velocity_z"],
                       help="Velocity z file relative to data-dir, without the suffix")
    muram.add_argument("--velocities", nargs="+", choices=AXES, default=list(AXES),
                       help="Velocity components to include; a synthesis needs "
                            "only the one along its line of sight")
    muram.add_argument("--header-file", type=str, default=MURAM_HEADER,
                       help="Header file relative to data-dir, without the suffix, "
                            "that the snapshot time is read from if it exists")
    muram.add_argument("--time", type=str, default=None,
                       help="Snapshot time with units, e.g. '26729.5 s', instead "
                            "of reading the header")
    muram.add_argument("--source", type=str, default="MURaM",
                       help="What to record as the source of the atmosphere")
    muram.add_argument("--compression", type=str, default=None,
                       help="h5py compression filter for the cubes, e.g. gzip; "
                            "the default writes them uncompressed")

    info = commands.add_parser("info", help="Describe an atmosphere file")
    info.add_argument("atmosphere", type=str, help="The file to describe")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        print(args.atmosphere)
        print(describe_atmosphere_file(args.atmosphere))
        return

    data_dir = Path(args.data_dir)
    output = args.output or f"{data_dir.resolve()}_{args.snapshot}.h5"
    files = {
        "temperature": args.temp_file,
        "mass_density": args.rho_file,
        "velocity_x": args.vx_file,
        "velocity_y": args.vy_file,
        "velocity_z": args.vz_file,
    }
    print(f"Reading MURaM snapshot {args.snapshot} from {data_dir}")
    atmosphere = read_muram(
        data_dir, args.snapshot, shape=args.cube_shape,
        voxel_dx=u.Quantity(args.voxel_dx), voxel_dy=u.Quantity(args.voxel_dy),
        voxel_dz=u.Quantity(args.voxel_dz), files=files,
        velocities=args.velocities, header=args.header_file,
        time=None if args.time is None else u.Quantity(args.time),
        source=args.source,
    )
    print(atmosphere.describe())
    if atmosphere.time is None:
        print("  No header file found, so the snapshot time is not recorded; "
              "give --time if a time series needs it.")
    write_atmosphere(atmosphere, output, compression=args.compression)
    print(f"Wrote {output} ({Path(output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
