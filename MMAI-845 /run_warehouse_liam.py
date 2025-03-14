from warehouse_liam import WarehouseEnv
import time

# Create the environment
env = WarehouseEnv(width=10, height=10, render_mode="human", num_items=5)

# Define action meanings
action_meanings = {
    0: "Move Left",
    1: "Move Right",
    2: "Move Up",
    3: "Move Down",
    4: "Pickup Item",
    5: "Drop Item"
}

# Reset the environment
observation, info = env.reset()

# Show initial state
print("\n=== INITIAL STATE ===")
env.render()

# Run a few random steps to see the environment in action
for step in range(20):
    # Take a random action
    action = env.action_space.sample()
    
    # Show the action about to be taken
    print(f"\n=== STEP {step+1} ===")
    print(f"Taking action: {action} ({action_meanings[action]})")
    
    # Step the environment
    observation, reward, done, truncated, info = env.step(action)
    
    # Show the result
    print(f"Result - Reward: {reward}")
    env.render()
    
    # Add a small delay to make it easier to follow
    time.sleep(0.5)
    
    # If the episode is done, reset the environment
    if done or truncated:
        print("\n=== ENVIRONMENT RESET ===")
        observation, info = env.reset()
        env.render()

# Close the environment
env.close()
