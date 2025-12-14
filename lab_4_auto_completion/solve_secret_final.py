"""
Solution for the secret task.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab_3_generate_by_ngrams.main import BeamSearcher, NGramLanguageModel
from lab_4_auto_completion.main import WordProcessor

def solve_secret() -> str:
    n_gram_size = 4
    beam_width = 6
    seq_len = 10
    
    secret_file_path = os.path.join(
        os.path.dirname(__file__), 
        "assets", 
        "secrets", 
        "secret_3.txt"
    )
    
    try:
        with open(secret_file_path, 'r', encoding='utf-8') as file:
            letter_text = file.read()
    except FileNotFoundError:
        return "Error: Secret file not found"
    
    if "<BURNED>" not in letter_text:
        return letter_text
    
    word_processor = WordProcessor("<EOS>")
    
    hp_text_path = os.path.join(os.path.dirname(__file__), "assets", "Harry_Potter.txt")
    additional_training_text = ""
    
    if os.path.exists(hp_text_path):
        try:
            with open(hp_text_path, 'r', encoding='utf-8') as file:
                hp_text = file.read()
            additional_training_text = hp_text
        except Exception:
            pass
    
    letter_without_burned = letter_text.replace("<BURNED>", "")
    training_text = additional_training_text + "\n\n" + letter_without_burned
    
    encoded_sentences = word_processor.encode_sentences(training_text)
    
    all_ids = []
    for sentence in encoded_sentences:
        all_ids.extend(sentence)
    
    if len(all_ids) == 0:
        return letter_text
    
    model = NGramLanguageModel(tuple(all_ids), n_gram_size)
    build_result = model.build()
    
    if build_result != 0:
        return letter_text
    
    searcher = BeamSearcher(beam_width, model)
    
    burned_pos = letter_text.find("<BURNED>")
    context_before = letter_text[max(0, burned_pos - 100):burned_pos]
    words_before = context_before.strip().split()
    start_words = []
    
    for word in reversed(words_before[-5:]):
        clean_word = ''.join(c for c in word if c.isalpha()).lower()
        if clean_word and len(clean_word) > 1:
            word_id = word_processor.get_id(clean_word)
            if word_id is not None:
                start_words.insert(0, word_id)
                if len(start_words) >= 2:
                    break
    
    if len(start_words) < 2:
        mother_id = word_processor.get_id("mother")
        know_id = word_processor.get_id("know")
        if mother_id is not None and know_id is not None:
            start_words = [mother_id, know_id]
        elif len(all_ids) >= 2:
            start_words = all_ids[-2:]
    
    start_sequence = tuple(start_words) if start_words else tuple(all_ids[-2:]) if len(all_ids) >= 2 else tuple(all_ids)
    
    candidates = {start_sequence: 0.0}
    
    for step in range(seq_len):
        new_candidates = {}
        
        for sequence, score in list(candidates.items()):
            next_tokens = searcher.get_next_token(sequence)
            
            if next_tokens is None or not next_tokens:
                new_candidates[sequence] = score
                continue
            
            updated = searcher.continue_sequence(sequence, next_tokens, {sequence: score})
            
            if updated:
                new_candidates.update(updated)
            else:
                new_candidates[sequence] = score
        
        if not new_candidates:
            break
        
        pruned = searcher.prune_sequence_candidates(new_candidates)
        if pruned is not None:
            candidates = pruned
        else:
            sorted_candidates = sorted(new_candidates.items(), key=lambda x: x[1])
            candidates = dict(sorted_candidates[:beam_width])
    
    if not candidates:
        best_sequence = start_sequence
    else:
        best_sequence = min(candidates.items(), key=lambda x: x[1])[0]
    
    decoded_words = []
    
    def get_word_by_id(word_id):
        if hasattr(word_processor, 'get_word'):
            word = word_processor.get_word(word_id)
            if word:
                return word
        
        if hasattr(word_processor, '_storage'):
            for w, wid in word_processor._storage.items():
                if wid == word_id:
                    return w
        
        return None
    
    for i, token_id in enumerate(best_sequence):
        word = get_word_by_id(token_id)
        
        if word and word != "<EOS>":
            decoded_words.append(word)
    
    if decoded_words:
        if len(decoded_words) >= 2 and decoded_words[0].lower() == "mother" and decoded_words[1].lower() == "know":
            generated_words = decoded_words[2:]
        else:
            generated_words = decoded_words
        
        generated_words = generated_words[:seq_len]
        final_generated_text = " ".join(generated_words)
    else:
        final_generated_text = ""
    
    completed_letter = letter_text.replace("<BURNED>", final_generated_text)
    
    return completed_letter

if __name__ == "__main__":
    completed = solve_secret()
    print(completed)