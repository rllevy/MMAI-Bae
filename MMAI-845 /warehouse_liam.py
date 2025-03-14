import gymnasium as gym
import numpy as np
from gymnasium import spaces

class Item:
    # Define a basic Item class with necessary attributes
    def __init__(self, item_id, position, shelf_position):
        self.id = item_id
        self.position = position  # Pickup position (in front of shelf)
        self.shelf_position = shelf_position  # Actual shelf position
        self.picked_up = False

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width), dtype=int)  # 0: empty, 2: shelf
        self.shelf_positions = []  # Track shelf positions
        
        # Add shelves
        for i in range(2, width-2, 3):
            for j in range(1, height-1):
                # Add shelf
                self.cells[j, i] = 2
                self.shelf_positions.append((i, j))
        
        # No obstacles for now
    
    def is_valid_position(self, position):
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            # Agent can move to empty cells (0), but not shelves (2)
            return self.cells[y, x] == 0
        return False

class WarehouseEnv(gym.Env):
    """Custom Warehouse Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, width=10, height=10, render_mode=None, num_items=5):
        super().__init__()
        self.render_mode = render_mode
        
        # Define size of the warehouse grid
        self.width = width
        self.height = height
        
        # Define action space
        # 0: left, 1: right, 2: forward, 3: backward, 4: pickup, 5: drop
        self.action_space = spaces.Discrete(6)
        
        # Define observation space
        # Agent position (x,y), carrying item (0/1), and grid representation
        self.observation_space = spaces.Dict({
            'agent_position': spaces.Box(low=0, high=max(width, height), shape=(2,), dtype=np.int32),
            'carrying_item': spaces.Discrete(2),
            'grid': spaces.Box(low=0, high=2, shape=(height, width), dtype=np.int32),
            'items': spaces.Box(low=0, high=max(width, height), shape=(num_items, 2), dtype=np.int32)
        })
        
        # Number of items to place in the warehouse
        self.num_items = num_items
        
        # Initialize the warehouse
        self.grid = None
        self.agent_position = None
        self.items = []
        self.carrying_item = None
        self.delivery_zone = None
        
        # Initialize counters
        self.steps_taken = 0
        self.items_delivered = 0
        self.max_steps = width * height * 4  # Reasonable step limit

    def step(self, action):
        reward = -0.1  # Small negative reward for each step to encourage efficiency
        done = False
        self.steps_taken += 1
        
        # Handle movement actions
        if action < 4:  # Movement actions
            new_position = self.agent_position.copy()
            
            if action == 0:  # Left
                new_position[0] = max(0, new_position[0] - 1)
            elif action == 1:  # Right
                new_position[0] = min(self.width - 1, new_position[0] + 1)
            elif action == 2:  # Forward (up)
                new_position[1] = max(0, new_position[1] - 1)
            elif action == 3:  # Backward (down)
                new_position[1] = min(self.height - 1, new_position[1] + 1)
            
            # Check if the new position is valid
            if self.grid.is_valid_position(new_position):
                self.agent_position = new_position
                
                # If carrying an item, update its position too
                if self.carrying_item is not None:
                    self.items[self.carrying_item].position = self.agent_position.copy()
        
        # Handle pickup action
        elif action == 4:  # Pickup
            if self.carrying_item is None:  # Not already carrying an item
                for i, item in enumerate(self.items):
                    if not item.picked_up and np.array_equal(item.position, self.agent_position):
                        self.carrying_item = i
                        item.picked_up = True
                        reward += 1.0  # Reward for picking up an item
                        break
        
        # Handle drop action
        elif action == 5:  # Drop
            if self.carrying_item is not None:  # Carrying an item
                # Check if in delivery zone
                if np.array_equal(self.agent_position, self.delivery_zone):
                    self.items_delivered += 1
                    reward += 10.0  # Big reward for delivering an item
                    self.items[self.carrying_item].position = [-1, -1]  # Remove from grid
                    self.carrying_item = None
                    
                    # Check if all items are delivered
                    if self.items_delivered == self.num_items:
                        done = True
                        reward += 50.0  # Bonus for completing the task
                else:
                    # Drop the item at the current position
                    self.carrying_item = None
                    reward -= 2.0  # Penalty for dropping outside delivery zone
        
        # Check if max steps reached
        if self.steps_taken >= self.max_steps:
            done = True
        
        info = {
            'items_delivered': self.items_delivered,
            'steps_taken': self.steps_taken
        }
        truncated = False
        return self._get_obs(), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset warehouse layout
        self.grid = Grid(self.width, self.height)
        
        # Reset agent position (start at bottom left)
        self.agent_position = np.array([0, self.height - 1])
        
        # Set delivery zone (top right)
        self.delivery_zone = np.array([self.width - 1, 0])
        
        # Reset items
        self.items = []
        
        # Get available shelf positions
        available_shelves = self.grid.shelf_positions.copy()
        np.random.shuffle(available_shelves)
        
        # Place items in front of shelves
        for i in range(min(self.num_items, len(available_shelves))):
            shelf_x, shelf_y = available_shelves[i]
            
            # Find the position in front of the shelf (to the left)
            pickup_x, pickup_y = shelf_x - 1, shelf_y
            
            # Make sure the pickup position is valid
            if 0 <= pickup_x < self.width and self.grid.cells[pickup_y, pickup_x] == 0:
                self.items.append(Item(i, np.array([pickup_x, pickup_y]), np.array([shelf_x, shelf_y])))
        
        # Reset state variables
        self.carrying_item = None
        self.steps_taken = 0
        self.items_delivered = 0
        
        info = {}
        return self._get_obs(), info

    def render(self):
        if self.render_mode == "human":
            # Create a visual representation of the grid
            grid_display = np.copy(self.grid.cells)
            
            # Mark agent position
            grid_display[self.agent_position[1], self.agent_position[0]] = 3
            
            # Mark items
            for item in self.items:
                if not item.picked_up and item.position[0] >= 0:  # Not picked up and not delivered
                    grid_display[item.position[1], item.position[0]] = 4
            
            # Mark delivery zone
            grid_display[self.delivery_zone[1], self.delivery_zone[0]] = 5
            
            print("\nWarehouse Grid:")
            print("' '=Empty, 'S'=Shelf, 'A'=Agent, 'I'=Item, 'D'=Delivery")
            print("-" * (self.width * 2 + 1))
            
            for i in range(self.height):
                row = "| "
                for j in range(self.width):
                    cell = int(grid_display[i, j])
                    symbol = " "
                    if cell == 2:
                        symbol = "S"  # Shelf
                    elif cell == 3:
                        symbol = "A"  # Agent
                    elif cell == 4:
                        symbol = "I"  # Item
                    elif cell == 5:
                        symbol = "D"  # Delivery
                    row += symbol + " "
                row += "|"
                print(row)
            
            print("-" * (self.width * 2 + 1))
            print(f"Carrying item: {self.carrying_item is not None}")
            print(f"Items delivered: {self.items_delivered}/{self.num_items}")
            print(f"Steps taken: {self.steps_taken}/{self.max_steps}")
            
        return self._get_obs()

    def _get_obs(self):
        # Create observation dictionary
        item_positions = np.array([item.position for item in self.items])
        return {
            'agent_position': self.agent_position,
            'carrying_item': 1 if self.carrying_item is not None else 0,
            'grid': self.grid.cells,
            'items': item_positions
        }

    def close(self):
        pass