import matplotlib.pyplot as plt
import pandas as pd
import json

# Leer el archivo JSON
df = pd.read_json('./training_rewards.json')

# Renombrar columnas correctamente
df_renamed = df.rename(columns={
    'iteration': 'Iteration',
    'reward_red_team': 'Red Team Reward',
    'reward_blue_team': 'Blue Team Reward'
})

print(df_renamed.columns)  # Verifica que ahora sí existen

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(df_renamed['Iteration'], df_renamed['Red Team Reward'], label='Red Team', color='red')
plt.plot(df_renamed['Iteration'], df_renamed['Blue Team Reward'], label='Blue Team', color='blue')

# Configuración del gráfico
plt.title('Reward per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
