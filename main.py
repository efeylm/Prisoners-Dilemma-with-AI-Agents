import random
import os
import time
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Define the Agent
class PrisonerAgent:
    def __init__(self, name, strategy, system_prompt=None):
        self.name = name
        self.strategy = strategy
        self.history = []
        self.opponent_history = []
        self.score = 0
        self.system_prompt = system_prompt

    def choose_action(self):
        return self.strategy(self)

    def update_history(self, my_action, opponent_action):
        self.history.append(my_action)
        self.opponent_history.append(opponent_action)

    def update_score(self, points):
        self.score += points

# OpenAI LLM strategy
def openai_strategy(agent):
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Using random choice as fallback.")
        return random.choice(["cooperate", "defect"])

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    round_num = len(agent.history) + 1

    # Build game history context
    history_text = ""
    if round_num > 1:
        history_text += "Game history:\n"
        for i in range(len(agent.history)):
            history_text += f"Round {i+1}: You {agent.history[i]}, Opponent {agent.opponent_history[i]}\n"

    system_prompt = agent.system_prompt or """
    You are an AI agent playing the Prisoner's Dilemma game. Choose either 'cooperate' or 'defect'.
    If both players cooperate, both get 3 points.
    If both players defect, both get 1 point.
    If one cooperates and one defects, the defector gets 5 points and the cooperator gets 0 points.
    Your goal is to maximize your total score.
    """

    user_prompt = f"""
    This is round {round_num} of the Prisoner's Dilemma game.

    {history_text}

    Based on this information, should you 'cooperate' or 'defect'? 
    Reply with exactly one word: either 'cooperate' or 'defect'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,
            temperature=0.5
        )
        response_text = response.choices[0].message.content.strip().lower()
        if "cooperate" in response_text:
            return "cooperate"
        elif "defect" in response_text:
            return "defect"
        else:
            return random.choice(["cooperate", "defect"])
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return random.choice(["cooperate", "defect"])

# Game logic
class PrisonersDilemma:
    def __init__(self, agent1, agent2, iterations=100, delay=0):
        self.agent1 = agent1
        self.agent2 = agent2
        self.iterations = iterations
        self.delay = delay
        self.payoff_matrix = {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1)
        }
        self.results = []

    def play_round(self):
        action1 = self.agent1.choose_action()
        action2 = self.agent2.choose_action()

        score1, score2 = self.payoff_matrix[(action1, action2)]

        self.agent1.update_history(action1, action2)
        self.agent2.update_history(action2, action1)
        self.agent1.update_score(score1)
        self.agent2.update_score(score2)

        self.results.append({
            'round': len(self.results) + 1,
            f'{self.agent1.name}_action': action1,
            f'{self.agent2.name}_action': action2,
            f'{self.agent1.name}_score': score1,
            f'{self.agent2.name}_score': score2
        })

        return action1, action2, score1, score2

    def run_simulation(self, verbose=False):
        for i in range(self.iterations):
            action1, action2, score1, score2 = self.play_round()

            if verbose:
                print(f"Round {i+1}: {self.agent1.name} {action1}, {self.agent2.name} {action2} → Scores: {score1}, {score2}")

            if self.delay > 0:
                time.sleep(self.delay)

        return self.results

    def get_summary(self):
        return {
            'iterations': self.iterations,
            f'{self.agent1.name}_total_score': self.agent1.score,
            f'{self.agent2.name}_total_score': self.agent2.score,
            f'{self.agent1.name}_cooperation_rate': self.agent1.history.count("cooperate") / len(self.agent1.history) if self.agent1.history else 0,
            f'{self.agent2.name}_cooperation_rate': self.agent2.history.count("cooperate") / len(self.agent2.history) if self.agent2.history else 0,
        }

# Simulation Example
if __name__ == "__main__":
    cooperative_prompt = """
    You are an AI agent playing the Prisoner's Dilemma. You value mutual benefit and long-term cooperation.
    Always try to build trust unless the opponent repeatedly defects.
    """

    competitive_prompt = """
    You are an AI agent playing the Prisoner's Dilemma. Your sole objective is to maximize your individual score,
    even if it means betraying the other player. Use strategic defection where beneficial.
    """

    agent1 = PrisonerAgent("CooperativeBot", openai_strategy, system_prompt=cooperative_prompt)
    agent2 = PrisonerAgent("CompetitiveBot", openai_strategy, system_prompt=competitive_prompt)

    game = PrisonersDilemma(agent1, agent2, iterations=10, delay=1)
    game.run_simulation(verbose=True)
    summary = game.get_summary()

    print("\nFinal Summary:")
    print(f"{agent1.name} scored {agent1.score}")
    print(f"{agent2.name} scored {agent2.score}")
    print(f"{agent1.name} cooperation rate: {summary[f'{agent1.name}_cooperation_rate']:.2f}")
    print(f"{agent2.name} cooperation rate: {summary[f'{agent2.name}_cooperation_rate']:.2f}")
