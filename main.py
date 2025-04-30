import gymnasium as gym
from numpy import zeros, argmax, max as npmax
from random import uniform
from time import sleep


class QlearningAgent:
    def __init__(self, alpha=0.95, gamma=0.5, epochs=1000):
        # Build the taxi environment
        self.env = gym.make("Taxi-v3") 
        self.learningRate = alpha  # learning rate
        self.discount = gamma  # discount factor
        self.max_steps = 100
        self.time = 1
        self.epochs = epochs
        # Initialize Q-Table and Nsa
        self.q_table = zeros([self.env.observation_space.n, self.env.action_space.n])
        self.nsa = zeros([self.env.observation_space.n, self.env.action_space.n])  # Frequency table for state-action pairs

    def choose_action(self, state, threshold=5):
        # Find actions with visitation counts below the threshold
        under_explored_actions = [action for action in range(self.env.action_space.n) if self.nsa[state, action] < threshold]
        
        if under_explored_actions:
            # Prioritize exploration of under-explored actions
            action = uniform(0, 1) < 1 / self.time and self.env.action_space.sample() or under_explored_actions[0]
        else:
            # Otherwise, choose the action with the highest Q-value
            action = argmax(self.q_table[state])  # Exploit learned values
        
        return action


    def step(self, action):
        new_state, reward, done, _, __ = self.env.step(action)
        return new_state, reward, done

    def update_q(self, state, new_state, reward, action):
        # Update frequency table
        self.nsa[state, action] += 1  # Increment count for this state-action pair
        # Retrieve old Q-value
        old_value = self.q_table[state, action]
        # Determine the best possible future value
        next_max = npmax(self.q_table[new_state])
        # Update Q-value using Q-learning formula
        new_value = old_value + self.learningRate * (reward + self.discount * next_max - old_value)
        self.q_table[state, action] = new_value

    def train(self):
        print('Training for {} epochs...'.format(self.epochs))
        for epoch in range(self.epochs):
            state = self.env.reset()[0]
            done = False
            steps = 0
            while not done and steps < self.max_steps:
                action = self.choose_action(state)
                new_state, reward, done = self.step(action)
                self.update_q(state, new_state, reward, action)
                state = new_state
                steps += 1
                self.time += 1

        self.env.close()

    def test(self):
        self.env = gym.make("Taxi-v3", render_mode="human")

        print('Test is started...')
        for epoch in range(5):
            state = self.env.reset()[0]
            done = False
            steps = 0
            while not done and steps < self.max_steps:
                sleep(0.3)
                action = argmax(self.q_table[state])  
                new_state, reward, done, _, __ = self.env.step(action)  # Take a step
                state = new_state
                steps += 1

        self.env.close()


if __name__ == "__main__":
    agent = QlearningAgent()
    agent.train()
    agent.test()
