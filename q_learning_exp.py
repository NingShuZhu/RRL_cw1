import numpy as np
from gymnasium.wrappers import TimeLimit
from my_cartpole_env import CartPoleEnv
from plotting import plot_returns, plot_returns_name
from collections import deque
from hyperparameters import *
from common import *


def run_q_learning_paras(env,train_timesteps_M,
                        alpha_schedule=False,
                        epsilon_schedule=False,
                        alpha_value=None):
    
    returns = deque(maxlen=1000)
    returns.append(0)
    saved_returns = []
    saved_timesteps = []
    episode = 0
    state, _ = env.reset()
    episode_reward = 0

    Q = np.zeros((N_BINS, N_BINS, N_BINS, N_BINS, 2))

    # Learning rate and exploration rate schedules flags
    # alpha_schedule = False
    # epsilon_schedule = False
    MIN_EPSILON = 1e-5
    MIN_ALPHA = 1e-5

    # print(f"Starting Q-learning with alpha_schedule={alpha_schedule}, epsilon_schedule={epsilon_schedule}, alpha_value={alpha_value}")

    for timestep in range(int(train_timesteps_M * 1e6)):
        
        if epsilon_schedule:
            ### FILL IN HERE ###
            # raise NotImplementedError("Epsilon schedule not implemented")
            # current_epsilon = EPSILON * (1 - episode / (train_timesteps_M * 1e6 / MAX_EPISODE_STEPS))
            total_timesteps = int(train_timesteps_M * 1e6)
            # current_epsilon = EPSILON * (1 - episode / total_timesteps)
            
            current_epsilon = max(
                MIN_EPSILON,
                EPSILON * (1 - timestep / total_timesteps)
            )
        else:
            current_epsilon = EPSILON

        if alpha_schedule:
            ### FILL IN HERE ###
            # raise NotImplementedError("Alpha schedule not implemented")
            # current_alpha = ALPHA * (1 - episode / (train_timesteps_M * 1e6 / MAX_EPISODE_STEPS))
            total_timesteps = int(train_timesteps_M * 1e6)
            # current_alpha = ALPHA * (1 - timestep / total_timesteps)
            
            current_alpha = max(
                MIN_ALPHA,
                ALPHA * (1 - timestep / total_timesteps)
            )
        else:
            # current_alpha = ALPHA
            base_alpha = alpha_value if alpha_value is not None else ALPHA
            current_alpha = base_alpha

        # current_epsilon = EPSILON
        # current_alpha = 0.03

        indices = state_to_indices(state, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
        
        ### FILL IN HERE ###
        # raise NotImplementedError("Action selection not implemented")
        if np.random.random() < current_epsilon:
            action = np.random.choice([0, 1]) # epsilon-greedy action selection
        else:
            action = np.argmax(Q[indices])

        next_state, _, terminated, truncated, _ = env.step(int(action))
        next_indices = state_to_indices(next_state, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
        reward = reward_function(next_state, terminated) * -1
        episode_reward += reward

        # Q-value update
        best_next_q = np.max(Q[next_indices])
        ### FILL IN HERE ###
        # raise NotImplementedError("Q-value update not implemented")
        Q[indices][action] += current_alpha * (reward + GAMMA * best_next_q - Q[indices][action])

        state = next_state
        if terminated or truncated:
            returns.append(episode_reward)
            episode += 1
            state, _ = env.reset()
            episode_reward = 0
      
        if timestep % 20000 == 0 and timestep > 0:
            # print(f"Timestep {timestep/1e6:.2f}M | Return: {np.mean(returns):.4f} | Eps: {current_epsilon:.5f} | Alpha: {current_alpha:.5f}")
            saved_returns.append(np.mean(returns))
            saved_timesteps.append(timestep/1e6)
            # plot_returns_name(saved_returns, saved_timesteps, filename=f"returns_{train_timesteps_M}M.png")

    return Q, saved_returns, saved_timesteps

def run_single_experiment(alpha_schedule,
                          epsilon_schedule,
                          alpha_value=None):

    print("======================================")
    print(f"alpha_schedule={alpha_schedule} | "
        f"epsilon_schedule={epsilon_schedule} | "
        f"alpha_value={alpha_value}")
    print("======================================")

    env = CartPoleEnv(
        x_threshold=X_LIMIT,
        theta_threshold_radians=THETA_LIMIT
    )
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    Q, saved_returns, saved_timesteps = run_q_learning_paras(
        env,
        train_timesteps_M=TRAIN_TIMESTEPS_M,
        alpha_schedule=alpha_schedule,
        epsilon_schedule=epsilon_schedule,
        alpha_value=alpha_value
    )

    if not alpha_schedule:
        filename = f"returns_alpha-s_{alpha_schedule}_epsilon-s_{epsilon_schedule}_alpha_{alpha_value}.png"
    else:
        filename = f"returns_alpha-s_{alpha_schedule}_epsilon-s_{epsilon_schedule}.png"

    plot_returns_name(saved_returns, saved_timesteps, filename=filename)

    # evaluate agent
    returns = evaluate_agent(env, type="Q", Q=Q)
    print_statistics(returns)

    env.close()


if __name__ == "__main__":

    # # train agent
    # train_env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    # train_env = TimeLimit(train_env, max_episode_steps=MAX_EPISODE_STEPS)
    # Q = run_q_learning_paras(train_env, train_timesteps_M=TRAIN_TIMESTEPS_M)

    # # evaluate agent
    # returns = evaluate_agent(train_env, type="Q", Q=Q)
    # print_statistics(returns)
    # train_env.close()

    # play agent
    # play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    # play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)
    # play_agent(play_env, type="Q", Q=Q)
    # play_env.close()

    # ========= 1️⃣ baseline：不同alpha =========
    for alpha in [0.1, 0.07, 0.05, 0.03, 0.01]:
        run_single_experiment(
            alpha_schedule=False,
            epsilon_schedule=False,
            alpha_value=alpha
        )

    # ========= 2️⃣ alpha schedule =========
    run_single_experiment(
        alpha_schedule=True,
        epsilon_schedule=False,
        alpha_value=None
    )

    # ========= 3️⃣ epsilon+alpha schedule =========
    run_single_experiment(
        alpha_schedule=True,
        epsilon_schedule=True,
        alpha_value=None
    )

    # ========= epsilon schedule =========
    run_single_experiment(
        alpha_schedule=False,
        epsilon_schedule=True,
        alpha_value=None
    )