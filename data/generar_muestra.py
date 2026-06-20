import pandas as pd
import numpy as np

# Configuración de la semilla para reproducibilidad
np.random.seed(42)

# 1. Definición de parámetros (Semillas extraídas de tus datos del SAIC)
# Formato: [Media Ingreso Mensual, Capital Inicial Promedio (X2), Proporción en el dataset]
sectores_config = {
    "46_Comercio": [291625, 732532, 0.40],  # 40% del dataset
    "72_Restaurantes": [172259, 1419803, 0.30], # 30% del dataset
    "81_Servicios": [44400, 267840, 0.30]   # 30% del dataset
}

n_total = 10000
data = []

# Función para generar valores log-normales basados en una media deseada
def log_normal_sample(mean_target, sigma, size):
    # Matemáticamente, para una log-normal, el promedio es exp(mu + sigma^2 / 2)
    mu_calc = np.log(mean_target) - (sigma**2 / 2)
    return np.random.lognormal(mu_calc, sigma, size)

# 2. Generación del Dataset
for sector, params in sectores_config.items():
    mu_ingreso, x2_capital, proporcion = params
    n_sector = int(n_total * proporcion)

    # X1: Sector (Codificado para el modelo)
    s_x1 = [sector] * n_sector

    # X2: Capital Inicial (Log-normal para realismo)
    s_x2 = log_normal_sample(x2_capital, 0.4, n_sector)

    # X3: Antigüedad (1 a 25 años)
    s_x3 = np.random.randint(1, 26, n_sector)

    # X4: Índice de Digitalización (0.0 a 1.0)
    s_x4 = np.random.uniform(0.1, 0.9, n_sector)

    # X5: Costos Operativos Mensuales (Relacionados al ingreso, aprox 70-85%)
    s_ingresos_temp = log_normal_sample(mu_ingreso, 0.5, n_sector)
    s_x5 = s_ingresos_temp * np.random.uniform(0.7, 0.9, n_sector)

    # X6: Inversión en Activos Fijos
    s_x6 = s_x2 * np.random.uniform(0.5, 0.8, n_sector)

    # X7: Acceso a Crédito (Binario 0 o 1)
    s_x7 = np.random.choice([0, 1], size=n_sector, p=[0.7, 0.3])

    # X8: Densidad de Competencia (Escala 1-10)
    s_x8 = np.random.randint(1, 11, n_sector)

    # X9: Capacidad de Producción (Porcentaje 0-100)
    s_x9 = np.random.uniform(40, 100, n_sector)

    # X10: Gasto en Marketing
    s_x10 = s_ingresos_temp * np.random.uniform(0.01, 0.08, n_sector)

    # --- VARIABLE OBJETIVO (Y) ---
    # Rentabilidad = (Ingresos - Costos - Marketing) / Ingresos
    s_y = ((s_ingresos_temp - s_x5 - s_x10) / s_ingresos_temp) * 100

    # Construcción de las filas
    for i in range(n_sector):
        data.append([
            s_x1[i], s_x2[i], s_x3[i], s_x4[i], s_x5[i],
            s_x6[i], s_x7[i], s_x8[i], s_x9[i], s_x10[i],
            s_y[i]
        ])

# 3. Creación del DataFrame y Guardado
columnas = [
    'X1_Sector', 'X2_Capital_Inicial', 'X3_Antiguedad', 'X4_Digitalizacion',
    'X5_Costos_Op', 'X6_Activos_Fijos', 'X7_Credito', 'X8_Competencia',
    'X9_Capacidad', 'X10_Marketing', 'Y_Rentabilidad'
]

df = pd.DataFrame(data, columns=columnas)

# Mezclar los datos para que no estén ordenados por sector
df = df.sample(frac=1).reset_index(drop=True)


print(f"Dataset generado exitosamente con {len(df)} registros.")
print(df.head())

# código literal, guarde el CSV así:
df.to_csv('data/dataset_tesis_final_corregido_9_6_26.csv', index=False)
