
def get_last_variable(ctx, name):
    for step in reversed(ctx["steps"]):
        if name in step:
            return step[name]

    return None
