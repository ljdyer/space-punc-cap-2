from tabulate import tabulate

# ====================
def tabulate_list_of_dicts(l):

    headers = {k: k for k in l[0].keys()}
    return tabulate(l, headers=headers)


# ====================
def get_dict_by_value(list_of_dicts, key, value):

    matches = [d for d in list_of_dicts if d[key] == value]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected a single match, but found {len(matches)} matches. " +\
            f"Key: {key}; Value: {value}"
        )
    return matches[0]