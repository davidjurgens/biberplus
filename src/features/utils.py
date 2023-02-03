def read_in_variables(txt_file):
    variables = []
    with open(txt_file, 'r') as f:
        for line in f:
            var = line.strip()
            if var:
                variables.append(var)
    return variables