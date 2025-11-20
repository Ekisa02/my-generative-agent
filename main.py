import time
from agents.reflect_llm import reflect_with_llm

state = {
    "name": "TinyAgent",
    "goals": ["learn", "help"],
    "memory": []
}

def observe():
    return "saw a tutorial about AI agents"

def reflect(state, observation):
    return reflect_with_llm(state["memory"], observation)

def decide_action(thought, state):
    if "learn" in thought:
        return "read more tutorials"
    return "rest"

def act(action, state):
    print(f"Action: {action}")
    state["memory"].append({"time": time.time(), "action": action})

for _ in range(3):
    obs = observe()
    print("Observe:", obs)

    th = reflect(state, obs)
    print("Reflect:", th)

    action = decide_action(th, state)
    act(action, state)

    time.sleep(1)

print("Final Memory:", state["memory"])
