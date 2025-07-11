# training_fixed.py
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
# Asumiendo que environment.py existe y es correcto
from environment import TronParallelEnv
import json
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import ray

def env_creator(_):
    return ParallelPettingZooEnv(TronParallelEnv())

register_env("tron_env", env_creator)

# Extrae espacios para definir las políticas
raw_env = TronParallelEnv()
# Es buena práctica cerrar el entorno después de usarlo
first_agent = raw_env.possible_agents[0]
obs_space = raw_env.observation_space[first_agent]
act_space = raw_env.action_space[first_agent]
raw_env.close()

# Definir las políticas por equipo
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id in ["player_1", "player_3"]:
        return "red_team_policy"
    else:
        return "blue_team_policy"

# Configurar PPO de la manera correcta
config = (
    PPOConfig()
    .environment("tron_env")
    .framework("torch")
    .rollouts(num_rollout_workers=1) # Para entrenamiento real, considera usar más workers si tienes CPUs
    .training(
        train_batch_size=2400,
        model={
            "fcnet_hiddens": [256, 256, 128],
            "fcnet_activation": "relu"
        },
        # Este es el parámetro correcto para controlar la exploración en PPO
        # Un valor más alto fomenta más exploración.
        
        
        entropy_coeff_schedule=[[0, 0.9], [140000, 0.05]]
    )
    # --- ELIMINADO EL BLOQUE .exploration(...) ---
    .multi_agent(
        policies={
            "red_team_policy": (None, obs_space, act_space, {}),
            "blue_team_policy": (None, obs_space, act_space, {})
        },
        policy_mapping_fn=policy_mapping_fn,
        # policies_to_train es opcional si quieres entrenar todas las políticas
        policies_to_train=["red_team_policy", "blue_team_policy"]
    )
)

# Iniciar Ray y entrenar
ray.init(ignore_reinit_error=True)

algorithm = config.build()

# Lista para guardar resultados
training_log = []

for i in range(210):
    results = algorithm.train()

    # Extraer recompensas medias de las políticas
    reward_red = results['policy_reward_mean'].get('red_team_policy', float('nan'))
    reward_blue = results['policy_reward_mean'].get('blue_team_policy', float('nan'))

    # Obtener la entropía de la política como métrica de exploración
    entropy_red = results['info']['learner']['red_team_policy']['learner_stats'].get('entropy', float('nan'))
    entropy_blue = results['info']['learner']['blue_team_policy']['learner_stats'].get('entropy', float('nan'))

    print(f"Iter {i}: reward_red={reward_red:.2f}, reward_blue={reward_blue:.2f}, "
          f"entropy_red={entropy_red:.2f}, entropy_blue={entropy_blue:.2f}")
          

    training_log.append({
        "iteration": i,
        "reward_red_team": reward_red,
        "reward_blue_team": reward_blue,
        "entropy_red": entropy_red,
        "entropy_blue": entropy_blue
    })

# Guardar resultados como JSON
with open("training_rewards.json", "w") as f:
    json.dump(training_log, f, indent=4)

# Guardar modelo entrenado
checkpoint_dir = algorithm.save("tron_ppo_2team")
print(f"Checkpoint saved in directory {checkpoint_dir}")

# Detener Ray
ray.shutdown()