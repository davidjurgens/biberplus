import stanza

stanza.download('en')
nlp = stanza.Pipeline('en')
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_dependencies()

def read_in_variables(txt_file):
    variables = []
    with open(txt_file, 'r') as f:
        for line in f:
            var = line.strip()
            if var:
                variables.append(var)
    return variables