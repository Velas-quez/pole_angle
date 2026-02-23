import torch
import pandas as pd
import numpy as np
import joblib
import mlflow.pytorch
import mlflow.artifacts
from os.path import join as pjoin

# --- 1. Configurações ---
RUN_ID = "c34ec37d74664b64ab931011d61488b0"  # Copie do dashboard do MLflow
TEST_FILE = pjoin("data", "test", "final_test_data_8quads.parquet")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_artifacts(run_id):
    print(f"Baixando artefatos da run: {run_id}...")

    # Carrega o modelo diretamente do MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(DEVICE)
    model.eval()

    # Baixa o scaler que salvamos como artefato
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessing/scaler.pkl"
    )
    scaler = joblib.load(scaler_path)

    return model, scaler


def evaluate(model, scaler, test_path):
    # 1. Carregar dados
    df = pd.read_parquet(test_path)

    # 2. SELEÇÃO RIGOROSA DE FEATURES
    # Filtramos apenas as colunas que começam com os prefixos táteis.
    # Isso garante que 'base_roll', 'source_bag', etc., fiquem de fora.
    feature_cols = [
        c for c in df.columns if any(p in c for p in ["mu_", "var_", "n_", "hu_"])
    ]

    print(feature_cols)
    print(scaler.n_features_in_)
    target_cols = ["target_x", "target_y", "target_z"]

    # IMPORTANTE: Verificamos se a quantidade de colunas bate com o scaler do treino
    if len(feature_cols) != scaler.n_features_in_:
        print(
            f"AVISO: O scaler espera {scaler.n_features_in_} features, mas foram encontradas {len(feature_cols)}."
        )
        # Se houver discrepância, forçamos a ordem exata baseada no que o scaler viu
        # (Isso assume que você não mudou os nomes das colunas entre treino e teste)
        if hasattr(scaler, "feature_names_in_"):
            feature_cols = scaler.feature_names_in_.tolist()
        else:
            raise ValueError(
                "O scaler não possui nomes de colunas e as dimensões não batem!"
            )

    X_raw = df[feature_cols].values
    y_true = df[target_cols].values

    # 3. Pré-processamento
    # Usamos o scaler original para transformar os dados de teste
    X_scaled = scaler.transform(X_raw)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(DEVICE)

    # 4. Inferência
    model.eval()
    with torch.no_grad():
        # A rede já normaliza a saída no forward() conforme definido anteriormente
        y_pred_tensor = model(X_tensor)

    y_pred = y_pred_tensor.cpu().numpy()
    y_pred[:, 1], y_pred[:, 2] = y_pred[:, 2].copy(), y_pred[:, 1].copy()

    # 5. Cálculo do Erro Angular (Métrica Real)
    # Cosine similarity: dot product de vetores unitários
    dot_product = np.sum(y_pred * y_true, axis=1)
    # Clip para evitar erros numéricos fora de [-1, 1] que quebram o arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Ângulo em graus: arccos(dot) * 180 / pi
    errors_deg = np.degrees(np.arccos(dot_product))

    # 6. Relatório de Resultados
    print("\n" + "=" * 30)
    print("   RESULTADOS DA AVALIAÇÃO")
    print("=" * 30)
    print(f"Número de Amostras:   {len(errors_deg)}")
    print(f"Erro Angular Médio:   {np.mean(errors_deg):.2f}°")
    print(f"Mediana do Erro:      {np.median(errors_deg):.2f}°")
    print(f"Desvio Padrão:        {np.std(errors_deg):.2f}°")
    print(f"Erro Máximo:          {np.max(errors_deg):.2f}°")
    print(f"MSE Vetorial:         {((y_true - y_pred) ** 2).mean():.6f}")
    print("=" * 30)

    # Verifique a norma e os eixos médios
    print(f"Média Real (X,Y,Z): {y_true.mean(axis=0)}")
    print(f"Média Predita (X,Y,Z): {y_pred.mean(axis=0)}")

    # Calcule o erro por eixo (MSE individual)
    for i, axis in enumerate(["X", "Y", "Z"]):
        error = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
        print(f"Erro MSE no eixo {axis}: {error:.6f}")

    return errors_deg, y_pred, y_true


if __name__ == "__main__":
    # Certifique-se que o tracking URI está correto se não for local
    # mlflow.set_tracking_uri("http://localhost:5000")

    m, s = load_artifacts(RUN_ID)
    evaluate(m, s, TEST_FILE)
