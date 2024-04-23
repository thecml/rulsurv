def get_window_size(n_condition: int):
    if n_condition == 0:
        return 2
    elif n_condition == 1:
        return 4
    elif n_condition == 2:
        return 6
    raise ValueError("N_condition not found")

def get_lag(n_condition: int):
    if n_condition == 0:
        return -1
    elif n_condition == 1:
        return -3
    elif n_condition == 2:
        return -5
    raise ValueError("N_condition not found")

def get_lmd(n_condition: int):
    if n_condition == 0:
        return 1.5
    elif n_condition == 1:
        return 1.75
    elif n_condition == 2:
        return 2
    raise ValueError("N_condition not found")