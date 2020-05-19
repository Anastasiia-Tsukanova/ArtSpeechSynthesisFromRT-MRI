# ArtSpeechSynthesisFromRT-MRI
A few scripts from my articulatory speech synthesis work

I used e-Lite HTS for phonetization of the corpus transcription, and the tool could not handle French diacritics despite being passed the input in UTF-8. It resulted in running the twice: once as it is to account for the syntax of the phrase (correct-synt), and the second time with the text where the spelling was transformed to produce correct pronunciation without using diacritics (correct-phon) - prepare_txt_for_e-lite.py. Then the two outputs were combined with Levenstein distance and modified for the use in Merlin (treat_e-lite_labels.py). 

visualize_art_params.py is used for visualizing articulatory parameter sequences for corpus sentences as well as for synthesized speech. test_000 and test_000.png are a sample output.
