
def flatten(iterable):
    for item in iterable:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def remove_duplicates(my_list: list) -> list:
    seen = set()
    duplicate_free = []

    for elem in my_list:
        if elem not in seen:
            duplicate_free.append(elem)
            seen.add(elem)

    return duplicate_free
