def load_word2vec(path):
    with open(path, 'r', encoding='utf-8') as f:
        word2vec = {}
        for line in f:
            temp = line.strip().split(' ')
            word = temp[0]
            vec = temp[1:]
            word2vec[word] = vec
    return word2vec