import sys
import os
from numba import njit
import argparse
import pandas as pd
import numpy as np
from plyfile import PlyData,PlyElement


def np_hash(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr



@njit
def jit_mode_reduceat(data, indices):
    """
    Computes the mode and the first index of that mode for sub-slices.

    Parameters:
    -----------
    data : ndarray
        The input array to process.
    indices : ndarray
        The start indices of each slice (similar to np.reduceat).
        Must be sorted.

    Returns:
    --------
    modes : ndarray
        The mode value for each slice.
    first_indices : ndarray
        The original index of the first occurrence of the mode.
    """
    n_slices = len(indices)
    modes = np.empty(n_slices, dtype=data.dtype)
    first_indices = np.empty(n_slices, dtype=np.int64)

    for i in range(n_slices):
        # Determine slice boundaries
        start = indices[i]
        end = indices[i + 1] if i < n_slices - 1 else len(data)

        if start >= end:
            # Handle empty slices if necessary
            continue

        # Extract segment and sort it to find the mode in O(N log N)
        # For very small ranges, a stack-based counter could be faster,
        # but sorting is robust for any value range.
        segment = data[start:end].copy()
        segment.sort()

        # Find the mode in the sorted segment
        current_mode_val = segment[0]
        max_count = 0

        tmp_count = 0
        tmp_val = segment[0]

        for j in range(len(segment)):
            if segment[j] == tmp_val:
                tmp_count += 1
            else:
                if tmp_count > max_count:
                    max_count = tmp_count
                    current_mode_val = tmp_val
                tmp_val = segment[j]
                tmp_count = 1

        # Final check for the last element group
        if tmp_count > max_count:
            current_mode_val = tmp_val

        modes[i] = current_mode_val

        # Find the first occurrence index in the original data
        for k in range(start, end):
            if data[k] == current_mode_val:
                first_indices[i] = k
                break

    return modes, first_indices

def downsample_pc(
    pc_df,
    grid_size=0.02,
    coord_cols=["x", "y", "z"],
    aggregation_func=dict(
        x="mean",
        y="mean",
        z="mean",
        red="mean",
        green="mean",
        blue="mean",
        scalar_label="mode",
        scalar_instance="mode_idx",
    ),
):
    outputs = {}

    scaled_coord = pc_df[coord_cols].values / np.array(grid_size)
    grid_coord = np.floor(scaled_coord).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord
    scaled_coord -= min_coord

    # Save the min coord in original values
    min_coord = min_coord * np.array(grid_size)

    # Hash of the grid coords -> to group the unique voxel coords
    key = np_hash(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]

    # unique values of the key
    # inverse: mapping from points to voxels (p2v_map)
    # count: points per voxel
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    # mapping from voxels to a single point (v2p_map)
    first_point_idx = idx_sort[np.cumsum(np.insert(count, 0, 0)[0:-1])]

    voxel_start_offsets = np.cumsum(np.insert(count, 0, 0)[0:-1])

    # Do mode mapping first
    mode_cols = [k for k in aggregation_func if aggregation_func[k] == "mode"]
    mode_idx_cols = [k for k in aggregation_func if aggregation_func[k] == "mode_idx"]
    if mode_idx_cols:
        assert len(mode_cols) == 1, "Only one mode var is supported if using mode_idx"

    # Calculate modes and indexes first to apply to mode_idx_cols later
    for col in mode_cols:
        values = pc_df[col].values[idx_sort]
        mode, mode_idxs = jit_mode_reduceat(values, voxel_start_offsets)
        outputs[col] = mode
    # Mode_idx_vars are those corresponding to the index of the first mode
    for col in mode_idx_cols:
        values = pc_df[col].values[idx_sort]
        outputs[col] = values[mode_idxs]

    for k in mode_cols:
        aggregation_func.pop(k)
    for k in mode_idx_cols:
        aggregation_func.pop(k)

    for col, agg_func in aggregation_func.items():
        values = pc_df[col].values
        output = None
        if agg_func == "first":
            output = values[first_point_idx]
        elif agg_func == "rand_choice":
            idx_select = idx_sort[
                voxel_start_offsets,
                +np.random.randint(0, count.max(), count.size) % count,
            ]
            output = values[idx_select]
        elif agg_func == "mean":
            output = (
                np.add.reduceat( values[idx_sort], voxel_start_offsets)
                / count
            )
        elif agg_func == "max":
            output = np.maximum.reduceat(values[idx_sort], voxel_start_offsets)
        elif agg_func == "min":
            output = np.minimum.reduceat( values[idx_sort], voxel_start_offsets)

        if output is not None:
            outputs[col] = output
        else:
            print(f"WARNING: column {col} yielded no output")

    return pd.DataFrame(outputs)

def save_as_ply(df, output_path):
    """
    Converts DataFrame back to a structured array and saves as a .ply file.
    """
    # Convert DataFrame to a list of tuples (required for structured arrays)
    records = df.to_records(index=False)
    
    # Create the PLY element (usually named 'vertex')
    el = PlyElement.describe(records, 'vertex')
    
    # Write to file
    PlyData([el]).write(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Read a .ply file, preprocess the data, and export to a file."
    )

    # Positional Argument
    parser.add_argument("input", help="Path to the source .ply file")

    # Optional Arguments
    parser.add_argument(
        "-o", "--outfile", 
        required=True, 
        help="Path for the processed output file (e.g., data.csv or data.parquet)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Display DataFrame head and info"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Source file '{args.input}' not found.")
        
    print(f"[*] Reading: {args.input}")
    plydata = PlyData.read(args.input)
    df = pd.DataFrame(plydata['vertex'].data)

    df = downsample_pc(df)

    # make sure that each instance has one and only one semantic label
    assert df.groupby("scalar_instance")['scalar_label'].nunique().max() == 1

    save_as_ply(df, args.outfile)

    print(f"[+] Processed data written to: {args.outfile}")

if __name__ == "__main__":
    main()