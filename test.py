token_length = 5
candidate_length = 4
levenshtein_matrix = []
levenshtein_matrix.append(list(range(candidate_length + 1)))
for i in range(1, token_length + 1):
    levenshtein_matrix_line = [i if ii == 0 else 0 for ii in range(candidate_length + 1)]
    levenshtein_matrix.append(levenshtein_matrix_line)
for i in levenshtein_matrix:
    print(i)