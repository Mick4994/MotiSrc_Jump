import numpy as np
import json
import os

class LineEnv:
    def __init__(self, target=5.0, max_steps=20):
        self.target = target      # 目标刻度
        self.max_steps = max_steps # 每轮最大步数
        self.action_space = [-1, 1] # 左右移动
        self.reset()
    
    def reset(self):
        self.position = 0.0       # 初始位置
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        return round(self.position, 1)  # 状态精确到小数点后1位
    
    def step(self, action):
        self.position += self.action_space[action]
        self.steps += 1
        
        done = (
            abs(self.position - self.target) < 0.1 or 
            self.steps >= self.max_steps
        )
        
        reward = -abs(self.position - self.target)  # 距离惩罚
        if abs(self.position - self.target) < 0.1:
            reward += 10  # 成功奖励
            
        return self._get_state(), reward, done

class QLearningAgent:
    def __init__(self, state_range, learning_rate=0.1, gamma=0.9):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = gamma
        self.state_range = state_range
    
    def get_action(self, state, epsilon=0.1):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
            
        if np.random.random() < epsilon:
            return np.random.choice(2)  # 随机探索
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
            
        self.q_table[state][action] += self.lr * (
            reward + 
            self.gamma * max(self.q_table[next_state]) - 
            self.q_table[state][action]
        )
    
    def save_model(self, filename):
        # 转换decimal为字符串键
        q_table_strkeys = {str(k):v for k,v in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(q_table_strkeys, f)

def train():
    env = LineEnv(target=5.0)
    agent = QLearningAgent(state_range=(-10, 10))
    
    episode_states = []
    
    for episode in range(200):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            # 记录每步状态（修复类型问题）
            episode_states.append({
                "episode": episode,
                "state": float(state),       # 转换为Python float
                "action": int(action),       # 转换为Python int
                "reward": float(reward)      # 转换为Python float
            })
            
            total_reward += reward
            state = next_state
            
            if done and episode % 20 == 0:
                # 修复保存时的数据转换
                os.makedirs("rl_result/line", exist_ok=True)
                with open(f"rl_result/line/episode_{episode}.json", 'w') as f:
                    json.dump(
                        [{
                            k: (float(v) if isinstance(v, (float, np.floating)) else v)
                            for k, v in item.items()
                        } for item in episode_states[-env.max_steps:]],
                        f, 
                        indent=2
                    )
                break
                
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.1f}")
    
    # 保存最终模型
    agent.save_model("q_learning_model.json")
    print("Training completed. Model saved.")

if __name__ == "__main__":
    train()