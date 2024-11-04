import numpy as np

# parameters
x = 0.25
y = 0.25
gamma = 0.9
threshold = 0.0001

V = np.zeros(4)
policy = ["nothing"]

def value_iteration(V, gamma, x, y):
    termination_rule = float("inf")
    while termination_rule >= threshold :
        termination_rule = 0
        V_prev = V.copy()

        # V*(S3)
        V[3] = 10 + gamma * V_prev[0]

        # V*(S2)
        V[2] = 1 + gamma * ((1 - y) * V_prev[0] + y * V_prev[3])

        # V*(S1)
        # V[1] = gamma * ((1 - x) * V_prev[1] + x * V_prev[3]) # Same than below
        V[1] = (gamma * x * V_prev[3])/(1-gamma*(1-x))

        # V*(S0)
        V0_a1 = gamma * V_prev[1]
        V0_a2 = gamma * V_prev[2]
        V[0] = max(V0_a1, V0_a2)

        policy[0] = "a1" if V0_a1 > V0_a2 else "a2"
        # Check for convergence
        termination_rule = np.max(np.abs(V - V_prev))
        
    for i in range(3):
        policy.append("a0")
    
    return V, policy

# Run value iteration
V_optimal, policy_optimal = value_iteration(V, gamma, x, y)

# Display results
print("Optimal values V* for each state:")
for i, v in enumerate(V_optimal):
    print(f"V*(S{i}) = {v:.4f}")

print("\nOptimal policy for each state:")
for i, p in enumerate(policy_optimal):
    print(f"pi*(S{i}) = {p}")