#
# render_custom.py
# 2DGS 自定义轨迹推理脚本 (适配 ACE / Scaffold-GS 逻辑)
#

import torch
import os
import numpy as np
import math
from argparse import ArgumentParser
from tqdm import tqdm

# 2DGS 核心模块
from gaussian_renderer import GaussianModel, render
from utils.mesh_utils import GaussianExtractor 
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import InferenceCamera  

# ==========================================
# 1. 7-Scenes 数据集硬编码配置
# ==========================================
SCENE_WIDTH = 640
SCENE_HEIGHT = 480

FOCAL_LENGTH_DICT = {
    'chess': 526.22, 
    'fire': 526.903, 
    'heads': 527.745, 
    'office': 525.143, 
    'pumpkin': 525.647, 
    'redkitchen': 525.505, 
    'stairs': 525.505
}

# ==========================================
# 2. 数学工具函数
# ==========================================
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def qvec2rotmat(qvec):
    """四元数 (w, x, y, z) 转 3x3 旋转矩阵"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2]])

def get_w2c_from_pose_ace(qvec, tvec):
    """
    【核心修正】适配 ACE / Scaffold-GS 的逻辑
    ACE 输出的通常已经是 W2C 格式 (COLMAP images.txt 格式)
    """
    # 1. 四元数转旋转矩阵
    R = qvec2rotmat(qvec)
    
    # 2. 【关键】完全复用 Scaffold-GS 的逻辑
    # Scaffold 代码: R = np.transpose(qvec2rotmat(qvec))
    # 3DGS 的 Camera 类会在内部再做一次 R.transpose()，所以这里先转置一次是正确的
    R_w2c = np.transpose(R)
    
    # 3. 【关键】Translation 直接用，不要求逆
    # Scaffold 代码: T = np.array(tvec)
    T_w2c = tvec
    
    return R_w2c, T_w2c

def load_custom_poses(txt_path):
    """读取 ACE 输出的位姿文件"""
    poses = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Pose file not found: {txt_path}")

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8: continue 
        
        # 解析 ACE 格式 (COLMAP images.txt 格式)
        # 格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # 或者: NAME, QW, QX, QY, QZ, TX, TY, TZ (取决于你的 txt 怎么生成的)
        
        # 假设你的 txt 格式是: Filename Qw Qx Qy Qz Tx Ty Tz ...
        name = parts[0]
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]) # w, x, y, z
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])]) # tx, ty, tz
        
        poses.append({'name': name, 'qvec': qvec, 'tvec': tvec})
    return poses

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    parser = ArgumentParser(description="Render ACE Poses for 2DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--pose_file", type=str, required=True, help="Path to the ACE poses.txt")
    parser.add_argument("--output_path", type=str, default="output_ace", help="Output directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Specific iteration to load")
    parser.add_argument("--scene_name", type=str, required=True, choices=FOCAL_LENGTH_DICT.keys(), help="7-Scenes sub-scene name")
    
    args = get_combined_args(parser)

    # ---------------------------------------
    # A. 准备相机参数
    # ---------------------------------------
    focal_length = FOCAL_LENGTH_DICT[args.scene_name]
    fov_y = focal2fov(focal_length, SCENE_HEIGHT)
    fov_x = focal2fov(focal_length, SCENE_WIDTH)
    print(f"[Config] Scene: {args.scene_name} | Focal: {focal_length}")

    # ---------------------------------------
    # B. 初始化模型
    # ---------------------------------------
    print("[1/4] Initializing Gaussian Model...")
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    
    if args.iteration == -1:
        search_path = os.path.join(args.model_path, "point_cloud")
        try:
            iters = [int(x.split('_')[-1]) for x in os.listdir(search_path) if x.startswith('iteration_')]
            if not iters: raise FileNotFoundError
            loaded_iter = max(iters)
            ply_path = os.path.join(search_path, f"iteration_{loaded_iter}", "point_cloud.ply")
        except:
            print(f"Error: No valid point cloud found in {args.model_path}")
            exit(1)
    else:
        ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")

    print(f"      Loading PLY from {ply_path}")
    gaussians.load_ply(ply_path)

    # ---------------------------------------
    # C. 初始化渲染器
    # ---------------------------------------
    print("[2/4] Initializing Extractor...")
    pipe = pipeline.extract(args)
    bg_color = [0, 0, 0] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    extractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    # ---------------------------------------
    # D. 构建相机列表 (ACE Logic)
    # ---------------------------------------
    print(f"[3/4] Loading ACE poses from {args.pose_file} ...")
    raw_poses = load_custom_poses(args.pose_file)
    camera_list = []
    file_names = []
    
    for item in raw_poses:
        # 使用修正后的函数，不做求逆，直接透传 W2C
        R_w2c, T_w2c = get_w2c_from_pose_ace(item['qvec'], item['tvec'])
        
        cam = InferenceCamera(
            image_name=item['name'],
            R=R_w2c,
            T=T_w2c,
            width=SCENE_WIDTH,
            height=SCENE_HEIGHT,
            fovx=fov_x,
            fovy=fov_y
        )
        camera_list.append(cam)
        file_names.append(item['name'])

    print(f"      Loaded {len(camera_list)} cameras.")

    # ---------------------------------------
    # E. 渲染与导出
    # ---------------------------------------
    print("[4/4] Reconstruction & Exporting...")
    
    # 渲染
    extractor.reconstruction(camera_list)
    
    # 导出 (RGB -> PNG, Depth -> NPY)
    # 使用你之前加到 utils/mesh_utils.py 里的函数
    extractor.export_image_custom(args.output_path)
    
    print("All Done!")