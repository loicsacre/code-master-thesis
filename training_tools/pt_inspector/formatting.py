def format_tree_view(generator, sep='.', vert='|  ', horz='|- '):
    prev_fix = []  # previous prefix
    for name, value in generator:
        name_parts = name.split(sep)

        i = 0
        while i < len(prev_fix) and i < len(name_parts) and \
              prev_fix[i] == name_parts[i]:
            i += 1

        n_vert = i-1
        for j in range(i, len(name_parts)):
            n_hz = 0 if j == 0 else 1
            v = value if j == len(name_parts)-1 else None

            yield '{}{}{}'.format(vert*n_vert, horz*n_hz, name_parts[j]), v
            n_vert += 1
        prev_fix = name_parts
