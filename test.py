text = "She is happy. He is happy"
text = text.lower()
tokens = []
current_word = []
for char in text:
    if char.isalpha():
        current_word.append(char)
    else:
        if current_word:
            tokens.extend(current_word)
            tokens.append('_')
            current_word = []
if current_word:
    tokens.extend(current_word)
    if text and (text[-1].isspace() or text[-1] in string.punctuation):
        tokens.append('_')
if not any(c.isalpha() for c in tokens):
    print('None')
print(tuple(tokens))