def calculate_average(numbers):
    if not numbers:
        raise ValueError("Cannot calculate average of an empty list")
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)


def get_user_name(user):
    if not isinstance(user, dict):
        raise TypeError(f"Expected dict, got {type(user).__name__}")
    name = user.get("name")
    if name is None:
        raise KeyError("User dict is missing required 'name' key")
    return str(name).upper()