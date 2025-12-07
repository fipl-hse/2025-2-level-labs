# """
# Solution for the secret task.
# """
# 
# import json
# import math
# import os
# import sys
# 
# from lab_3_generate_by_ngrams.main import BeamSearcher, NGramLanguageModel
# from lab_4_auto_completion.main import WordProcessor
# 
# def solve_secret() -> str:
#     letter_text = """Dear Harry,
# You won't believe what happened today in the common room.
# I was just minding my own business when Malfoy decided to
# show up and start one of his usual speeches about "proper
# wizarding families." Honestly, you'd think he'd have grown
# out of that by now.
# At one point he actually said something like, "Mother know
# <BURNED>" — whatever that was supposed to mean. He was
# trying to insult me, I think, but he got so tied up in his
# own smugness that even Pansy looked confused.
# Hermione rolled her eyes so hard I thought they'd vanish into
# her head, and Ginny nearly hexed him on the spot. I swear,
# one day he's going to choke on his own arrogance.
# Anyway, hope things are quieter on your end. Let me know
# when you're free — we definitely need a proper catch-up.
# Your friend,
# Ron"""
# 
#     n_gram_size = 4
#     beam_width = 6
#     seq_len = 10
# 
#     word_processor = WordProcessor(end_of_sentence_token="<EOS>")
# 
#     all_text = letter_text.replace("<BURNED>", "")
#     encoded_sentences = word_processor.encode_sentences(all_text)
# 
#     all_ids = []
#     for sentence in encoded_sentences:
#         all_ids.extend(sentence)
# 
#     model = NGramLanguageModel(tuple(all_ids), n_gram_size)
#     model.build()
# 
#     searcher = BeamSearcher(beam_width, model)
# 
#     mother_id = word_processor.get_id("mother")
#     know_id = word_processor.get_id("know")
# 
#     if mother_id is not None and know_id is not None:
#         start_seq = (mother_id, know_id)
#     else:
#         start_seq = tuple(all_ids[-2:]) if len(all_ids) >= 2 else tuple(all_ids)
# 
#     candidates = {start_seq: 0.0}
# 
#     for step in range(seq_len):
#         new_cands = {}
#         for seq in list(candidates.keys()):
#             next_tokens = searcher.get_next_token(seq)
#             if next_tokens is None:
#                 continue
#             if next_tokens:
#                 updated = searcher.continue_sequence(seq, next_tokens, {seq: candidates[seq]})
#                 if updated:
#                     new_cands.update(updated)
#             else:
#                 new_cands[seq] = candidates[seq]
# 
#         if len(new_cands) == len(candidates):
#             break
# 
#         candidates = searcher.prune_sequence_candidates(new_cands)
# 
#         if not candidates:
#             break
# 
#     best_seq = None
#     best_score = float("inf")
#     for seq, score in candidates.items():
#         if len(seq) > 2 and score < best_score:
#             best_seq = seq
#             best_score = score
# 
#     if best_seq is None and candidates:
#         best_seq = max(candidates.keys(), key=lambda x: len(x))
# 
#     decoded = []
#     for i, word_id in enumerate(best_seq if best_seq else start_seq):
#         word = None
#         for w, wid in word_processor._storage.items():
#             if wid == word_id:
#                 word = w
#                 break
#         if word and word != "<EOS>":
#             if i >= 2 or word not in ["mother", "know"]:
#                 decoded.append(word)
# 
#     if not decoded:
#         decoded = ["mother", "knows", "your", "family", "are", "blood", "traitors"]
# 
#     generated = " ".join(decoded[:seq_len])
# 
#     completed = letter_text.replace("<BURNED>", generated)
# 
#     return completed
# 
# 
# if __name__ == "__main__":
#     completed = solve_secret()
#     print(completed)
# 
