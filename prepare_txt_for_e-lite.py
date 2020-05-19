# -*- coding: utf-8 -*-

# Doesn't run from Komodo for some reason. conda's environment: textprocessing

import sys
from nltk.tokenize import word_tokenize
import re
import os

def ensure_dir(directory):
    if directory != '':
        if (not os.path.exists(directory)) or (not os.path.isdir(directory)):
            os.makedirs(directory)

def transform(file_path_original, file_path_with_correct_pron, file_path_with_correct_syntax):
    with open(file_path_original, encoding='utf-8') as file_original:
        text_original = file_original.read()
        ensure_dir(os.path.dirname(file_path_with_correct_syntax))
        with open(file_path_with_correct_syntax, "w+", encoding='utf-8') as file_with_correct_syntax:
            # Treat corpus-specific annotation
            text_with_correct_syntax = text_original.replace('--', '.')
            text_with_correct_syntax = text_with_correct_syntax.replace('_', ' ')
            text_with_correct_syntax = re.sub(r'{\w+}', '', text_with_correct_syntax)
            words = [w.lower() for w in word_tokenize(text_with_correct_syntax, language='french')]
            for k, word in enumerate(words):
                # Treat corpus-specific annotation
                words[k] = re.sub(r'-$', '', words[k])
                # Treat special characters
                words[k] = words[k].replace(':', '')
                words[k] = words[k].replace('*', '')
                words[k] = words[k].replace('?', '.')
                words[k] = words[k].replace('!', '.')
                words[k] = words[k].replace('(', '').replace(')', '')
                # Basic changes over, saving the result to file_with_correct_syntax
                file_with_correct_syntax.write(re.sub(r'\s+', ' ', " ".join(words)))
                # Treat apostrophes
                words[k] = re.sub(r'^([ctlsn])\'(es)(t?)$', r'\1ai', re.sub(r'^c\'', 's\'', words[k])).replace('\'', '')
                # Treat é
                pref_e_acc_aigu_s_vowel = re.compile("^[rd]és[aouiyeéèêàûôîùëïüœ]")
                e_acc_aigu_s_vowel = re.compile("és[aouiyeéèêàûôîùëïüœ]")
                if e_acc_aigu_s_vowel.match(words[k]):
                    if not pref_e_acc_aigu_s_vowel.match(words[k]):
                        e_aigu_s_V_replacement = r'er z\1'
                    else:
                        e_aigu_s_V_replacement = r'er s\1'
                    words[k] = re.sub(r'és([aouiyeéèêàûôîùëïüœ])', e_aigu_s_V_replacement, words[k])
                words[k] = re.sub(r'ée$', 'er ', words[k]).replace('é', 'er ')
                # Treat accent grave
                words[k] = words[k].replace('è', 'ai')
                words[k] = words[k].replace('à', 'a')
                words[k] = words[k].replace('ù', 'u')
                # Treat ë, ï, ü
                words[k] = re.sub(r'ë$', 'e', words[k]).replace('ë', 'ai')    # ambiguë -> ambigue, Noël -> Noail
                words[k] = words[k].replace('ï', ' i')                        # naïve -> na ive
                words[k] = words[k].replace('güe', 'gu').replace('ü', 'u')
                words[k] = words[k].replace('ö', 'oe')
                words[k] = words[k].replace('ä', 'ai')
                # Treat accent circonflexe â, ê, î, ô, û
                words[k] = words[k].replace('â', 'a')
                words[k] = words[k].replace('ê', 'ai')
                words[k] = words[k].replace('û', 'u')
                words[k] = words[k].replace('ô', 'o')
                words[k] = words[k].replace('î', 'i')
                # Treat ç
                words[k] = words[k].replace('ç', 's')
                # Treat ligatures
                words[k] = words[k].replace('æ', 'ae')
                words[k] = words[k].replace('œ', 'e')
                # Extra
                words[k] = words[k].replace('ñ', 'gn')
            ensure_dir(os.path.dirname(file_path_with_correct_pron))
            with open(file_path_with_correct_pron, "w+", encoding='utf-8') as file_with_correct_pron:
                file_with_correct_pron.write(re.sub(r'\s+', ' ', " ".join(words)))

if __name__ == "__main__":

    if len(sys.argv)<3:
        print('Usage: python prepare_txt_for_e-lite.py <original_text_file> <file_with_wrong_vocab_and_syntax_but_correct_pron> <file_with_correct_vocab_and_syntax_but_wrong_pron>\n')
        sys.exit(0)

    file_path_original = sys.argv[1]
    file_path_with_correct_pron  = sys.argv[2]
    file_path_with_correct_syntax   = sys.argv[3]

    transform(file_path_original, file_path_with_correct_pron, file_path_with_correct_syntax)
            
        
                
            

