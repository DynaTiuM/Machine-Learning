from arrow_environment import ArrowEnvironment

initial_state = [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]

def q_iteration(env, gamma=0.9, theta=1e-3):
    n = env.n
    q_table = {}

    for state in range(2 ** n):
        state_list = [int(x) for x in bin(state)[2:].zfill(n)]
        actions = env.available_actions(state_list)
        q_table[tuple(state_list)] = {action: 0 for action in actions}
    
    q_table_update = {}

    while True:
        delta = 0
        for state in q_table:
            actions = env.available_actions(list(state))
            for action in actions:
                old_value = q_table[state][action]
                next_state = env.transition(list(state), action)
                max_q_next = max(q_table[tuple(next_state)].values(), default=0)
                reward = 100 if env.is_goal_state(next_state, initial_state) else 0
                q_table_update[(state, action)] = reward + gamma * max_q_next
                
                delta = max(delta, abs(old_value - q_table_update[(state, action)]))

        for (state, action), new_value in q_table_update.items():
            q_table[state][action] = new_value
        

        if delta < theta:
            break
        
    return q_table

def extract_optimal_policy(q_table):
    policy = {}
    for state, actions in q_table.items():
        if actions:
            best_action = max(actions, key=actions.get)
            policy[state] = best_action
    return policy

env = ArrowEnvironment(initial_state)

q_table = q_iteration(env)
optimal_policy = extract_optimal_policy(q_table)
#print("Politique optimale:", optimal_policy)

action_sequence = env.get_action_sequence(initial_state, optimal_policy)
print("Séquence d'actions:", action_sequence)