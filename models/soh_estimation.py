def capacity_fade(cycle_count, initial_capacity):
    fade_rate = 0.01  # 1% capacity loss per 100 cycles
    current_capacity = initial_capacity * (1 - fade_rate * cycle_count / 100)
    soh = current_capacity / initial_capacity
    return max(soh, 0)