class ArrowEnvironment:
    def __init__(self, initial_state):
        self.n = len(initial_state)

    def transition(self, state, action):
        new_state = state.copy()
        
        if action[0] == 1:
            i = action[1]
            new_state[i:i+3] = [1 - x for x in new_state[i:i+3]]
        
        elif action[0] == 2:
            i = action[1]
        
            new_state[i] = 1 - state[i]
            new_state[i+1] = 1 - state[i+1]
    
        return new_state
    
    def available_actions(self, state):
        actions = []

        for i in range(self.n - 2):
            if len(set(state[i:i+3])) == 1:
                actions.append((1, i))

        for i in range(self.n - 1):
            if state[i] != state[i+1]:
                actions.append((2, i))

        return actions
    
    def is_goal_state(self, state, initial_state):
        return state == [1 - arrow for arrow in initial_state]
    
    def get_action_sequence(self, initial_state, policy):
        state = initial_state
        action_sequence = []

        while not self.is_goal_state(state, initial_state):
            action = policy.get(tuple(state))
            if action is None:
                break
            
            action_sequence.append(action)
            next_state = self.transition(state, action)
            state = next_state

        return action_sequence
