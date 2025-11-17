# change_detection_regions.py
# Single-file change detection via ICP + geometry-only region growing.
# Dependencies: open3d, numpy, pandas, laspy, scipy

import os
import time
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

# ----------------------------
# I/O: load point cloud files
# ----------------------------
def load_point_cloud(fp: str) -> o3d.geometry.PointCloud:
    """
    Load .las/.laz using laspy (requires lazrs for .laz), or .ply via Open3D.
    Returns an Open3D PointCloud with XYZ only.
    """
    fp = os.path.abspath(fp)
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    ext = os.path.splitext(fp.lower())[1]
    if ext in [".las", ".laz"]:
        import laspy  # lazy import so script fails early if missing
        las = laspy.read(fp)
        pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(fp)
        if pcd.is_empty():
            raise ValueError(f"PLY at {fp} loaded empty.")
        return pcd
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .las, .laz, or .ply")


# --------------------------------
# Preprocess (downsample + normals)
# --------------------------------
def preprocess_pcd(pcd: o3d.geometry.PointCloud,
                   voxel: float = 0.10,
                   remove_outliers: bool = True) -> o3d.geometry.PointCloud:
    """
    Voxel downsample, estimate normals, and optionally remove statistical outliers.
    """
    q = pcd.voxel_down_sample(voxel)
    if len(q.points) == 0:
        raise ValueError("Downsampling produced 0 points; try a smaller voxel.")
    q.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 3.0, max_nn=50
        )
    )
    if remove_outliers:
        q, _ = q.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return q


# -----------------------
# Bounding-box quick look
# -----------------------
def report_bbox(pcd: o3d.geometry.PointCloud, name: str = "pcd"):
    arr = np.asarray(pcd.points)
    if arr.size == 0:
        print(f"[{name}] EMPTY")
        return
    center = arr.mean(0)
    extents = arr.max(0) - arr.min(0)
    print(f"[{name}] n={len(arr)}  center={center.round(3)}  extents={extents.round(3)}")


# --------------------
# ICP (point-to-plane)
# --------------------
def align_icp_point_to_plane(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             voxel: float = 0.10,
                             max_iter: int = 60) -> tuple[o3d.geometry.PointCloud, np.ndarray, float, float]:
    """
    Align source -> target using point-to-plane ICP.
    Returns (source_aligned, 4x4, fitness, rmse).
    """
    # make copies (avoid in-place)
    s = o3d.geometry.PointCloud(source)
    t = o3d.geometry.PointCloud(target)

    # ensure normals
    if not s.has_normals():
        s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=50))
    if not t.has_normals():
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=50))

    # correspondence threshold ~ 2 * voxel
    corr_thresh = voxel * 2.0
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter
    )
    result = o3d.pipelines.registration.registration_icp(
        s, t, corr_thresh,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    s.transform(result.transformation)
    return s, result.transformation, float(result.fitness), float(result.inlier_rmse)


# ---------------------------------------------
# NN distances (t1_aligned points → nearest t2)
# ---------------------------------------------
def nn_distances(source_aligned: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud) -> np.ndarray:
    """
    For each point in source_aligned, compute distance to nearest neighbor in target.
    """
    src = np.asarray(source_aligned.points)
    tgt = np.asarray(target.points)
    if len(src) == 0 or len(tgt) == 0:
        raise ValueError("Empty point cloud(s) for NN distances.")
    tree = KDTree(tgt)
    dists, _ = tree.query(src, k=1)
    return dists.astype(np.float64)


# --------------------------------
# Distance heatmap (blue→red PLY)
# --------------------------------
def make_distance_heatmap_pcd(source_aligned: o3d.geometry.PointCloud,
                              distances: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Color source_aligned by distances (blue=small, red=large).
    """
    heat = o3d.geometry.PointCloud(source_aligned)
    if len(distances) != len(heat.points):
        raise ValueError("distances length != number of points")
    mn, mx = float(np.min(distances)), float(np.max(distances))
    if mx <= mn:
        norm = np.zeros_like(distances, dtype=np.float64)
    else:
        norm = (distances - mn) / (mx - mn)
    colors = np.zeros((len(norm), 3), dtype=np.float64)
    colors[:, 0] = norm              # red up with distance
    colors[:, 2] = 1.0 - norm        # blue down with distance
    colors[:, 1] = np.where(norm < 0.5, norm * 2.0, (1.0 - norm) * 2.0)  # green for mid
    heat.colors = o3d.utility.Vector3dVector(colors)
    return heat


# ----------------------------------------------------
# Missing-mask + region-growing (pure geometry, no QSM)
# ----------------------------------------------------
def find_missing_indices(distances: np.ndarray, nn_threshold: float) -> np.ndarray:
    """
    Indices in source_aligned where NN distance exceeds threshold (potentially 'missing' in t2).
    """
    return np.where(distances > nn_threshold)[0].astype(np.int64)


def detect_missing_regions(source_aligned: o3d.geometry.PointCloud,
                           missing_idx: np.ndarray,
                           neighbor_radius: float = 0.05,
                           min_region_pts: int = 30) -> list[np.ndarray]:
    """
    Region-growing over *missing* points only.
    Returns a list of arrays of indices INTO missing_idx (not global indices).
    """
    if missing_idx is None or len(missing_idx) == 0:
        return []

    src = np.asarray(source_aligned.points)
    miss = src[missing_idx]
    tree = KDTree(miss)
    visited = np.zeros(len(miss), dtype=bool)
    regions: list[np.ndarray] = []

    for i in range(len(miss)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        reg = [i]
        while stack:
            j = stack.pop()
            nbrs = tree.query_ball_point(miss[j], r=neighbor_radius)
            for n in nbrs:
                if not visited[n]:
                    visited[n] = True
                    stack.append(n)
                    reg.append(n)
        if len(reg) >= min_region_pts:
            regions.append(np.asarray(reg, dtype=int))

    return regions


# ---------------------------
# PCA axis / length / angles
# ---------------------------
def pca_axis(pts: np.ndarray):
    """
    Returns (unit_axis, centroid, projections_along_axis, length_along_axis).
    """
    C = pts.mean(axis=0)
    X = pts - C
    # SVD for principal axis
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = Vt[0]
    if axis[2] < 0:
        axis = -axis
    proj = X @ axis
    L = float(max(0.0, proj.max() - proj.min()))
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis, C, proj, L


def summarize_regions(source_aligned: o3d.geometry.PointCloud,
                      missing_idx: np.ndarray,
                      regions: list[np.ndarray]) -> pd.DataFrame:
    """
    Compute per-region summary: size, centroid, PCA length, bbox, a crude radius proxy,
    azimuth/elevation of the axis.
    """
    cols = [
        "region_id","n_points","centroid_x","centroid_y","centroid_z",
        "length_m","bbox_x","bbox_y","bbox_z","radius_med_m","azimuth_deg","elevation_deg"
    ]
    if not regions:
        return pd.DataFrame(columns=cols)

    src = np.asarray(source_aligned.points)
    rows = []
    for rid, reg in enumerate(regions, start=1):
        pts = src[missing_idx[reg]]
        axis, C, proj, L = pca_axis(pts)
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        ext = pmax - pmin
        # median distance to axis as crude thickness estimate
        V = pts - C
        r_med = float(np.median(np.linalg.norm(np.cross(V, axis), axis=1)))
        # angles
        az = float(np.degrees(np.arctan2(axis[1], axis[0])) % 360.0)
        el = float(np.degrees(np.arctan2(axis[2], np.linalg.norm(axis[:2]) + 1e-12)))

        rows.append({
            "region_id": rid,
            "n_points": int(len(pts)),
            "centroid_x": float(C[0]),
            "centroid_y": float(C[1]),
            "centroid_z": float(C[2]),
            "length_m": float(L),
            "bbox_x": float(ext[0]),
            "bbox_y": float(ext[1]),
            "bbox_z": float(ext[2]),
            "radius_med_m": r_med,
            "azimuth_deg": az,
            "elevation_deg": el,
        })

    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["n_points", "length_m"], ascending=[False, False]).reset_index(drop=True)


# -----------------------------------------
# Visualization / export (PLY + CSV + logs)
# -----------------------------------------
def colorize_regions_full(source_aligned: o3d.geometry.PointCloud,
                          missing_idx: np.ndarray,
                          regions: list[np.ndarray],
                          unchanged_color=(0.7, 0.7, 0.7)) -> o3d.geometry.PointCloud:
    """
    Color full source cloud: unchanged grey; each region gets a unique color.
    """
    N = len(source_aligned.points)
    colors = np.tile(np.asarray(unchanged_color, float), (N, 1))
    base = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.6, 0.2, 0.8],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)
    for k, reg in enumerate(regions):
        c = base[k % len(base)]
        colors[missing_idx[reg]] = c

    out = o3d.geometry.PointCloud(source_aligned)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out


def export_regions_artifacts(out_dir: str,
                             t1_aligned: o3d.geometry.PointCloud,
                             distances: np.ndarray,
                             missing_idx: np.ndarray,
                             regions: list[np.ndarray],
                             regions_df: pd.DataFrame):
    """
    Write:
      - distance_heatmap.ply (t1 colored by NN distance)
      - missing_regions_colored_full.ply (grey + colored regions)
      - missing_regions_points.ply (concatenated region points only, red)
      - missing_regions_summary.csv
      - change_stats.csv (basic counts)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Heatmap
    heat = make_distance_heatmap_pcd(t1_aligned, distances)
    o3d.io.write_point_cloud(os.path.join(out_dir, "distance_heatmap.ply"), heat)

    # Colored full
    colored = colorize_regions_full(t1_aligned, missing_idx, regions)
    o3d.io.write_point_cloud(os.path.join(out_dir, "missing_regions_colored_full.ply"), colored)

    # Region points only
    if regions:
        src = np.asarray(t1_aligned.points)
        all_pts = np.concatenate([src[missing_idx[r]] for r in regions], axis=0)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(all_pts)
        pc.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.io.write_point_cloud(os.path.join(out_dir, "missing_regions_points.ply"), pc)

    # CSV
    regions_df.to_csv(os.path.join(out_dir, "missing_regions_summary.csv"), index=False)

    # Basic stats
    stats = dict(
        n_t1=len(t1_aligned.points),
        nn_threshold_used=float(NN_THRESHOLD),
        n_missing_total=int(len(missing_idx)),
        n_regions=int(len(regions)),
        mean_missing_dist=float(np.mean(distances[missing_idx])) if len(missing_idx) else 0.0,
        max_missing_dist=float(np.max(distances[missing_idx])) if len(missing_idx) else 0.0
    )
    pd.DataFrame([stats]).to_csv(os.path.join(out_dir, "change_stats.csv"), index=False)


# -----------
# Main runner
# -----------
def run_change_detection(t1_path: str,
                         t2_path: str,
                         out_dir: str,
                         voxel: float = 0.10,
                         nn_threshold: float = 0.15,
                         neighbor_radius: float = 0.05,
                         min_region_pts: int = 30,
                         visualize: bool = False):
    """
    Full pipeline: load → preprocess → align → NN distances → missing mask →
    region growing → summaries → exports.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    print("\n== Load ==")
    pcd1 = load_point_cloud(t1_path)
    pcd2 = load_point_cloud(t2_path)
    report_bbox(pcd1, "t1_raw")
    report_bbox(pcd2, "t2_raw")

    print("\n== Preprocess ==")
    t1 = preprocess_pcd(pcd1, voxel=voxel, remove_outliers=True)
    t2 = preprocess_pcd(pcd2, voxel=voxel, remove_outliers=True)
    report_bbox(t1, "t1_pre")
    report_bbox(t2, "t2_pre")

    print("\n== ICP align (t1 → t2) ==")
    t1_aligned, T, fit, rmse = align_icp_point_to_plane(t1, t2, voxel=voxel, max_iter=60)
    print(f"ICP: fitness={fit:.4f}, rmse={rmse:.4f}")

    print("\n== NN distances & missing mask ==")
    dists = nn_distances(t1_aligned, t2)
    missing_idx = find_missing_indices(dists, nn_threshold=nn_threshold)
    print(f"Missing points above {nn_threshold:.3f} m: {len(missing_idx)} / {len(dists)}")

    print("\n== Region growing on missing points ==")
    regions = detect_missing_regions(
        t1_aligned, missing_idx,
        neighbor_radius=neighbor_radius,
        min_region_pts=min_region_pts
    )
    print(f"Detected {len(regions)} regions (r={neighbor_radius:.3f}, min_pts={min_region_pts})")

    print("\n== Region summaries ==")
    regions_df = summarize_regions(t1_aligned, missing_idx, regions)
    print(regions_df.head())

    print("\n== Export artifacts ==")
    export_regions_artifacts(out_dir, t1_aligned, dists, missing_idx, regions, regions_df)

    if visualize:
        print("\n== Interactive visualization ==")
        # Distance heatmap
        heat = make_distance_heatmap_pcd(t1_aligned, dists)
        # Colored regions
        colored = colorize_regions_full(t1_aligned, missing_idx, regions)
        # Show both with t2 (context)
        t2_vis = o3d.geometry.PointCloud(t2)
        t2_vis.paint_uniform_color([0.6, 0.8, 1.0])
        o3d.visualization.draw_geometries([t2_vis, heat])
        o3d.visualization.draw_geometries([t2_vis, colored])

    print(f"\nDone in {time.time() - t0:.2f}s. Results -> {os.path.abspath(out_dir)}")
    return {
        "t1_aligned": t1_aligned,
        "distances": dists,
        "missing_idx": missing_idx,
        "regions": regions,
        "regions_df": regions_df
    }


# =================
# USER SETTINGS
# =================
if __name__ == "__main__":
    # Input clouds (epoch-1 vs epoch-2)
    T1_PATH = "./data/arbre2022_26918.laz"   # change me
    T2_PATH = "./data/arbre2023_26918.laz"   # change me

    # Output directory for all artifacts
    OUT_DIR = "./results/26918_0001_regions_only"

    # Processing parameters
    VOXEL         = 0.10   # m, for downsample + ICP search radii
    NN_THRESHOLD  = 0.15   # m, flag “missing” if t1→t2 NN distance > this
    NEIGHBOR_R    = 0.05   # m, region-grow radius in missing points
    MIN_REGION_N  = 50     # min pts to keep a region

    # Visualization (Open3D windows)
    VISUALIZE = False

    run_change_detection(
        t1_path=T1_PATH,
        t2_path=T2_PATH,
        out_dir=OUT_DIR,
        voxel=VOXEL,
        nn_threshold=NN_THRESHOLD,
        neighbor_radius=NEIGHBOR_R,
        min_region_pts=MIN_REGION_N,
        visualize=VISUALIZE
    )
