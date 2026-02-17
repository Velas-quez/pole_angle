import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os


def fix_preprocessing_files(input_markers, input_opti, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df_markers = pd.read_parquet(input_markers)
    df_opti = pd.read_parquet(input_opti)

    print(f"Processando {len(df_markers)} linhas...")

    # --- 1. CORREÇÃO DOS MARKERS ---
    # Remapeamento: X -> X | Y -> Z | Z -> -Y (para o Z ficar vertical positivo)
    def rotate_points(pts_list):
        if pts_list is None or len(pts_list) == 0:
            return pts_list
        pts = np.array(pts_list).reshape(-1, 3)

        new_pts = np.zeros_like(pts)
        new_pts[:, 0] = pts[:, 0]  # X mantém
        new_pts[:, 1] = pts[:, 2]  # Novo Y recebe antigo Z
        new_pts[:, 2] = -pts[:, 1]  # Novo Z recebe antigo -Y (Vertical Up)

        return new_pts.flatten().tolist()

    if "markers" in df_markers.columns:
        df_markers["markers"] = df_markers["markers"].apply(rotate_points)
        print("-> Marcadores remapeados para Z-Up.")

    # --- 2. CORREÇÃO DA ORIENTAÇÃO (QUATERNIONS) ---
    # Precisamos de uma rotação que leve o referencial da base para o novo sistema.
    # Essa rotação é de 90 graus no eixo X.
    rot_fix = R.from_euler("x", 90, degrees=True)

    def rotate_quat(q_dict):
        # Converte dict para SciPy Rotation (ordem x,y,z,w)
        q_orig = R.from_quat([q_dict["x"], q_dict["y"], q_dict["z"], q_dict["w"]])
        # Aplica a rotação de correção
        q_new = rot_fix * q_orig
        quat = q_new.as_quat()
        return {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]}

    if "orientation" in df_opti.columns:
        df_opti["orientation"] = df_opti["orientation"].apply(rotate_quat)
        print("-> Quaternions da base corrigidos.")

    # 3. Salvar
    m_out = os.path.join(output_folder, "markers2_fixed.parquet")
    o_out = os.path.join(output_folder, "optitrack2_fixed.parquet")

    df_markers.to_parquet(m_out, index=False)
    df_opti.to_parquet(o_out, index=False)

    print(f"\nFeito! Use os arquivos em '{output_folder}' para gerar o dataset final.")


if __name__ == "__main__":
    fix_preprocessing_files(
        input_markers="data/test/preprocessed/markers2.parquet",
        input_opti="data/test/preprocessed/optitrack2.parquet",
        output_folder="data/test/fixed",
    )
