def linear_find_x(x_list, y_list, target_y, is_required_y = False):
    target_x = None

    n = len(x_list)
    if n == 0: return target_x
    if y_list[0] > target_y: return target_x
    if y_list[0] == target_y: return y_list[0]
    if y_list[-1] == target_y: return y_list[1]
    
    ith = 0
    while ith < n -1:
        if y_list[ith] <= target_y and y_list[ith+1] > target_y:
            target_x = x_list[ith] + (target_y-y_list[ith])*(x_list[ith]-x_list[ith+1])/(y_list[ith]-y_list[ith+1])
            break
        ith += 1
    
    if is_required_y:
        return target_x, target_y
    else:
        return target_x

def nearest_find_x_at_least(x_list, y_list, target_y):
    target_x = None
    new_target_y = None

    n = len(x_list)
    if n == 0: return target_x
    if y_list[0] > target_y: return target_x
    if y_list[0] == target_y: return y_list[0]
    if y_list[-1] == target_y: return y_list[1]
    
    ith = 0
    while ith < n-1:
        if y_list[ith] < target_y and y_list[ith+1] >= target_y:
            target_x = x_list[ith+1]
            new_target_y = y_list[ith+1]
            break
        ith += 1
    return target_x, new_target_y
