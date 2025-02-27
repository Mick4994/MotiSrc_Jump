import pygame
import numpy as np
import random
from pygame.locals import *

# 测试 [img_comments/rl_test/test.jpg]

# 环境参数优化
class EnhancedTriangleEnv:
    def __init__(self, width=400, height=300, grid_size=40):  # 缩小窗口尺寸和网格
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.target = (width//2, height//2)
        self.triangle_size = 30
        
        # 状态空间维度计算
        self.x_states = width // grid_size
        self.y_states = height // grid_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Enhanced Triangle RL")

    def reset(self):
        """重置环境时限制初始位置范围"""
        self.pos = [
            random.randint(self.width//4, 3*self.width//4),
            random.randint(self.height//4, 3*self.height//4)
        ]
        return self._get_state()

    def _get_state(self):
        """添加相对位置特征"""
        dx = (self.target[0] - self.pos[0]) // self.grid_size
        dy = (self.target[1] - self.pos[1]) // self.grid_size
        return (dx, dy)

    def _calculate_reward(self, new_pos):
        """改进的奖励函数"""
        distance = np.hypot(new_pos[0]-self.target[0], new_pos[1]-self.target[1])
        
        reward = 0
        # 方向性奖励（鼓励向目标移动）
        prev_distance = np.hypot(self.pos[0]-self.target[0], self.pos[1]-self.target[1])
        if distance < prev_distance:
            reward += 2  # 靠近奖励
        elif distance > prev_distance:
            reward -= 1  # 远离惩罚
            
        # 成功奖励（放宽成功条件）
        if distance < self.grid_size * 1.5:
            reward += 50
            if distance < 5:  # 精确对齐奖励
                reward += 100
                
        # 边界惩罚（降低惩罚力度）
        if (new_pos[0] < 0 or new_pos[0] >= self.width or
            new_pos[1] < 0 or new_pos[1] >= self.height):
            reward -= 2
            
        return reward

    def step(self, action):
        """添加动作惯性机制"""
        move_dict = {
            0: (0, -self.grid_size//2),  # 上（半格移动）
            1: (0, self.grid_size//2),   # 下
            2: (-self.grid_size//2, 0), # 左
            3: (self.grid_size//2, 0)    # 右
        }
        dx, dy = move_dict[action]
        new_pos = [self.pos[0]+dx, self.pos[1]+dy]
        
        # 边界限制（允许部分越界回弹）
        new_pos[0] = np.clip(new_pos[0], 0, self.width)
        new_pos[1] = np.clip(new_pos[1], 0, self.height)
        
        self.pos = new_pos
        done = np.hypot(*self._get_state()) < 1  # 基于相对位置的终止条件
        return self._get_state(), self._calculate_reward(new_pos), done

    def render(self):
        """添加可视化方法"""
        self.screen.fill((255, 255, 255))  # 白色背景
        
        # 绘制目标点
        pygame.draw.circle(self.screen, (255,0,0), self.target, 8)
        
        # 绘制移动三角形（用等腰三角形表示方向）
        triangle_points = [
            (self.pos[0], self.pos[1] - self.triangle_size//2),
            (self.pos[0] - self.triangle_size//2, self.pos[1] + self.triangle_size//2),
            (self.pos[0] + self.triangle_size//2, self.pos[1] + self.triangle_size//2)
        ]
        pygame.draw.polygon(self.screen, (0,128,0), triangle_points)
        
        pygame.display.update()

# 改进的Q-learning智能体
class EnhancedQLAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros(state_size + (action_size,))
        self.alpha = 0.5    # 增大学习率
        self.gamma = 0.9     # 降低折扣因子
        self.epsilon = 0.7   # 初始高探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                                      self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
        # epsilon衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 训练流程优化
def enhanced_train():
    env = EnhancedTriangleEnv(grid_size=40)
    
    # 根据状态空间维度初始化Q表
    state_dims = (
        (env.width//env.grid_size)*2 + 1,  # dx范围 [-n, n]
        (env.height//env.grid_size)*2 + 1  # dy范围
    )
    agent = EnhancedQLAgent(state_dims, 4)
    
    episodes = 2000  # 增加训练次数
    success_count = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:  # 限制最大步数
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # # 每50次渲染一次加速训练
            # if episode % 50 == 0:
            env.render()
            pygame.time.wait(10)
                
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                    
        if done:
            success_count += 1
            
        # 每100轮显示进度
        if episode % 100 == 0:
            print(f"Episode {episode+1}: Success rate {success_count/100:.0%}")
            success_count = 0

if __name__ == "__main__":
    enhanced_train()
    pygame.quit()