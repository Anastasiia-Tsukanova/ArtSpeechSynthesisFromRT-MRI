import sys,os
from shutil import copy2 as cpy
import codecs

separators = ["^", "-", "+", "=", "@", "/"] # Watch out, there is schwa encoded as @
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
translate = {'a': 'a', 'l': 'l', 'O': 'oopen', 'R': 'r', '_': 'sil', 'pau': 'sil', 'e': 'e', 'z': 'z', 'v': 'v', 'a~': 'an', 't': 't', 'Z': 'zh', 'e~': 'en', 'k': 'k', 'o~': 'on', 'n': 'n', 'j': 'j', 'J': 'gn', 'd': 'd', 'b': 'b', 'i': 'i', 'p': 'p', 'm': 'm', 'E': 'eps', '@': 'schwa', 'y': 'y', 'o': 'o', 'u': 'u', 's': 's', '2': 'deux', 'S': 'sh', 'f': 'f', 'w': 'w', '9~': 'oen', 'H': 'h', 'g': 'g', '9': 'oe', 'N': 'nn', 'x': 'x'}
vowels = ['a', 'O', 'e', 'a~', 'e~', 'o~', 'i', 'E', '@', 'y', 'o', 'u', '2', '9~', '9', 'w', 'H', 'j', '_']

def parse(label, counter):
    # p1^p2- p3+p4=p5 @p6 p7/A:a1 a2 a3 /B:b1-b2-b3 @b4-b5 &b6-b7 #b8-b9 $b10-b11
    # !b12-b13 ;b14-b15 |b16 //C:c1+c2+c3/D:d1 d2 /E:e1+e2 @e3+e4 &e5+e6 #e7+e8
    # /F: f1 f2/G:g1 g2 /H:h1=h2 @h3=h4|h5 /I:i1 i2/J: j1+ j2- j3

    # 0 500000 x^x-a+l=O@1_1/A:0_0_/B:1-1-1@1-2&1-2#0-1$0-1!0-1;0-1|a/C:1+1+3/D:x_0/E:ADV+2@1+1&0+0#0+1/F:LIGHTPUNCT-1/G:0_0/H:2=1@0=6|NONE/I:1_1/J:20+12-7
    if " " in label:
        t0 = label.find(" ")
        prefix = label[:label.find(" ", t0+1)]
        no_invented_time_stamps = True
    else:
        no_invented_time_stamps = False
        prefix = "{} {}".format(100000*counter, 100000*(counter+1)) # Add false duration information
    phonemes = list()
    prev_idx = len(prefix) if no_invented_time_stamps else -1
    for sep in separators[:-2]:
        phonemes.append(label[prev_idx+1:label.find(sep)])
        prev_idx = label.find(sep)
    no_ph_beyond = label.find(separators[-1])
    phonemes.append(label[prev_idx+1:label[:no_ph_beyond].rfind(separators[-2])])
    prev_idx = label[:no_ph_beyond].rfind(separators[-2])
    rest = label[prev_idx+1:]
    return prefix, phonemes, rest

def parse_rest(label_rest):
    constituents = list()
    for prev_cat, next_cat in zip([None] + categories, categories + [None]):
        if prev_cat is None:
            left_idx, right_idx = 0, label_rest.find('/'+next_cat+':')
        elif next_cat is None:
            left_idx, right_idx = label_rest.find('/'+prev_cat+':')+3, len(label_rest)
        else:
            left_idx, right_idx = label_rest.find('/'+prev_cat+':')+3, label_rest.find('/'+next_cat+':')
        constituents.append(label_rest[left_idx:right_idx])
    return constituents

def treat_anticipated_vowel(rest):
    rest_transl = rest
    for vowel in vowels:
        rest_transl = rest_transl.replace("|%s/C:" % vowel, "|%s/C:" % translate[vowel])
    return rest_transl

def encode(prefix, phonemes, rest):
    label = prefix + " " if prefix else ''
    for ph, sep in zip(phonemes, separators):
        if ph in translate:
            label += translate[ph] + sep
        elif ph in translate.values():
            label += ph + sep
        else:
            print("Unknown phoneme %s" % ph)
    return label+treat_anticipated_vowel(rest)

def make_concise(labels):
    conc_labels = list()
    same_lab_since = 0
    for curr_line, next_line in zip(labels, labels[1:]):
        curr_t_start, curr_t_end, curr_rest = curr_line.split(' ')
        # print(curr_t_start + ' +++ '+  curr_t_end + ' +++ '+  curr_rest)
        next_t_start, next_t_end, next_rest = next_line.split(' ')
        curr_t_start, curr_t_end, next_t_start, next_t_end = map(lambda x: int(x), [curr_t_start, curr_t_end, next_t_start, next_t_end])
        def cut(line, symb):
            cut_line = line.split(symb)
            phonemes, label = cut_line[0], symb.join(cut_line[1:])
            try:
                num = int(label[label.rfind('[')+1:label.rfind(']')])
            except:
                num = None
            return phonemes, label[:label.rfind('[')], num
        curr_phonemes, curr_label, _ = cut(curr_rest, '@')
        next_phonemes, next_label, _ = cut(next_rest, '@')
        # if '/E:ENDPUNCT+' in curr_label and '/E:ENDPUNCT+' not in next_label:
        #     midpoint = min(curr_t_start, int(0.5*(same_lab_since + curr_t_end)))
        #     conc_labels.append("{} {} {}@{}\n".format(coef*curr_t_start, coef*midpoint, curr_phonemes, curr_label))
        #     conc_labels.append("{} {} {}@{}\n".format(coef*midpoint, coef*curr_t_end, curr_phonemes, curr_label))
        #     same_lab_since = next_t_start
        if curr_phonemes != next_phonemes or curr_label != next_label:
            conc_labels.append("{} {} {}@{}\n".format(same_lab_since, curr_t_end, curr_phonemes, curr_label))
            same_lab_since = next_t_start
    conc_labels.append("{} {} {}@{}".format(same_lab_since, next_t_end, next_phonemes, next_label))
    return conc_labels

def seq_matcher(reference, to_match, costs = (2, 2, 1)):
    # Matching to_match against reference. Output: [0, 1, 2, 2, 3] means that elt#2 needs to be copied for a match to happen; [0, 1, 3] means that el#2 needs to be skipped.
    # tolerance refers to how many elements may be skipped or how many copies 
    matched_idx, curr_idx = list(), len(to_match)-1
    s, t = list(to_match), list(reference)
    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    for row in range(1, rows):
        dist[row][0] = row * deletes
    for col in range(1, cols):
        dist[0][col] = col * inserts
    for col in range(1, cols):
        for row in range(1, rows):
            dist[row][col] = min(dist[row-1][col] + deletes, dist[row][col-1] + inserts, dist[row-1][col-1] + (0 if s[row-1] == t[col-1] else substitutes))
            # action == 0 -> s transforms into t through a deletion, action == 1 -> through an insertion, == 2 -> a substitution
    row, col = rows-1, cols-1
    big_number = rows*cols*max(costs)
    while (row, col) != (0, 0):
        substitute_cost = (dist[row-1][col-1] - dist[row][col]) if row > 0 and col > 0 else big_number
        deletion_cost = (dist[row-1][col] - dist[row][col]) if row > 0 and col >= 0 else big_number
        insertion_cost = (dist[row][col-1] - dist[row][col]) if row >= 0 and col > 0 else big_number
        if substitute_cost <= deletion_cost and substitute_cost <= insertion_cost:
            matched_idx = [curr_idx] + matched_idx
            row, col = row-1, col-1
            curr_idx -= 1
        elif deletion_cost <= substitute_cost and deletion_cost <= insertion_cost:
            row, col = row-1, col
            curr_idx -= 1
        else: # insertion_cost <= deletion_cost and insertion_cost <= substitute_cost:
            matched_idx = [curr_idx] + matched_idx
            row, col = row, col-1
    assert len(matched_idx) == len(reference)
    return matched_idx


def match_label_lists(labels_from_phon, labels_from_synt):
    labels_from_phon_parsed = [parse(label_ph, counter+1) if len(label_ph) > 1 else None for counter, label_ph in enumerate(labels_from_phon)][:-1]
    labels_from_synt_parsed = [parse(label_synt, counter+1) if len(label_synt) > 1 else None for counter, label_synt in enumerate(labels_from_synt)][:-1]
    if None in labels_from_phon_parsed:
        if None not in labels_from_synt_parsed:
            return labels_from_synt_parsed, labels_from_synt_parsed
        else:
            return list(), list()
    if None in labels_from_synt_parsed:
        return labels_from_phon_parsed, labels_from_phon_parsed
    # The resulting length will be len(labels_from_phon_parsed) - we need to find counterparts of labels_from_phon_parsed in labels_from_synt_parsed
    # There may be cases when multiple phonetically correct labels correspond to a single syntax labels - in this case we need to insert copies of that syntax label
    # There may be cases when a single phonetic label corresponds to multiple syntax labels - in this case we keep only one
    ph_ls_from_phon = [label[1][2] for label in labels_from_phon_parsed]
    ph_ls_from_synt = [label[1][2] for label in labels_from_synt_parsed]
    return labels_from_phon_parsed, [labels_from_synt_parsed[idx] for idx in seq_matcher(ph_ls_from_phon, ph_ls_from_synt)]

def encode_silence(label_from_phon, label_from_synt):
    pref_from_phon, phs_from_phon, _ = label_from_phon
    _, _, rest_from_synt = label_from_synt
    fix_next_time = False
    t0_orig, t1_orig = [int(t) for t in pref_from_phon.split(" ")]
    if t0_orig == 0:
        fix_next_time = True
        pref_sil = "0 %s" % str(round(float(t1_orig)/2))
    else:
        pref_sil = "0 %d" % t0_orig
    phs_sil = ['x', 'x', 'sil', phs_from_phon[2], phs_from_phon[3]]
    rest_from_synt_constits = parse_rest(rest_from_synt)
    def sc(component_name, left_sep, right_sep, split_val='-', which_elt=1):
        component = rest_from_synt_constits[categories.index(component_name)+1]
        section = component[component.find(left_sep)+1 if left_sep is not None else 0:component.find(right_sep) if right_sep is not None else len(component)]
        return section.split(split_val)[which_elt]
    rest_sil_constits = ['1_1', # 1 elt in the current syll from the beginning, 1 from the end
                         '0_0_', # A: the previous syllable is not stressed neither accented, no phonemes in it
                         '0-0-1@0-%s&0-%s#0-%s$0-%s!0-%s;0-%s|%s' % (sc('B', '@', '&'), sc('B', '@', '&'), sc('B', '#', '$'), sc('B', '$', '!'), sc('B', '!', ';'), sc('B', ';', '|'), sc('B', '|', 'None', which_elt=0)), # B: whether the current syllable is stressed or not, how many,
                         rest_from_synt_constits[3], # C: next syll stressed
                         'x_0', # D: POS prev
                         'x+%s@0+0&0+%s#0+%s' % (sc('E', None, '@', '+'), sc('E', '&', '#', '+'), sc('E', '#', None, '+')), # E: POS content curr
                         rest_from_synt_constits[6], # F: POS next
                         '0_0', # G: syll words prev
                         rest_from_synt_constits[8], # H: syll words pos curr
                         rest_from_synt_constits[9], # I: syll words next
                         rest_from_synt_constits[10]]
    rest_sil = rest_sil_constits[0]
    for part, constituent in enumerate(rest_sil_constits[1:]):
        rest_sil += '/%s:' % str(categories[part]) + constituent
    return encode(pref_sil, phs_sil, rest_sil), fix_next_time

if __name__ == "__main__":

    if len(sys.argv)<3:
        print('Usage: python treat_e-lite_labels.py <e-lite_correct_phon_dir> <e-lite_correct_synt_dir> <e-lite_labels_corrected_lab_dir>\n')
        sys.exit(0)

    elite_corr_phon_dir   = sys.argv[1]
    elite_corr_synt_dir   = sys.argv[2]
    elite_out_dir  = sys.argv[3]

    # Treat Gottingen data:
    # wav_dir = os.path.join(elite_in_dir, '..', 'gottingen_wav')
    # scp_file = os.path.join(elite_in_dir, '..', 'file_id_list.scp')
    # def remove_underscores(directory):
    #     for dp, dn, fn in os.walk(directory):
    #         for f in fn:
    #             os.rename(os.path.join(dp, f), os.path.join(dp, f.replace("_", ""))) 
    # remove_underscores(elite_in_dir)
    # remove_underscores(wav_dir)
    # with open(scp_file, 'r') as scp_f:
    #     file_list = scp_f.read()
    # with open(scp_file, 'w') as scp_f:
    #     scp_f.write(file_list.replace("_", ""))

    speakers = [sp for sp in os.listdir(elite_corr_phon_dir) if os.path.isdir(os.path.join(elite_corr_phon_dir,sp))]
    if speakers == list():
        speakers = ['']
    for sp in speakers:
        if not os.path.exists(os.path.join(elite_out_dir,sp)):
            os.makedirs(os.path.join(elite_out_dir,sp))
        for f_phon_raw in os.listdir(os.path.join(elite_corr_phon_dir,sp)):
            f_conv = f_phon_raw[:-4].replace('-corr-phon', '')+'.lab'
            # cpy(os.path.join(elite_corr_phon_dir,sp,f_phon_raw), os.path.join(elite_out_dir,sp,f_conv))
            with open(os.path.join(elite_corr_phon_dir,sp,f_phon_raw), 'r') as f_before_fixing_synt:
                print("treat_e-lite_labels applied to %s..." % os.path.join(elite_out_dir,sp,f_conv))
                labels_from_phon = f_before_fixing_synt.readlines()
                with open(os.path.join(elite_corr_synt_dir,sp,f_phon_raw[:-4].replace('-corr-phon', '-corr-synt')+'.lab'), 'r') as f_with_corr_synt:
                    labels_from_synt = f_with_corr_synt.readlines()
                with open(os.path.join(elite_out_dir,sp,f_conv), 'w+') as f_after_fixing_syntax:
                    labels_from_phon, labels_from_synt = match_label_lists(labels_from_phon, labels_from_synt)
                    if labels_from_phon == list():
                        continue
                    label_silence, fx_time = encode_silence(labels_from_phon[0], labels_from_synt[0])
                    labels_out = [label_silence]
                    for (pref_from_phon, phs_from_phon, _), (_, _, rest_from_synt) in zip(labels_from_phon, labels_from_synt):
                        labels_out.append(encode(pref_from_phon, phs_from_phon, rest_from_synt))
                    labels_out[1] = labels_out[1].replace('x^x-', 'x^sil-')
                    if fx_time:
                        init_lab_spl = labels_out[1].split(' ')
                        t0_orig, t1_orig = init_lab_spl[0], init_lab_spl[1]
                        labels_out[1] = labels_out[1].replace('%s %s ' % (t0_orig, t1_orig), '%d %s ' % (round(float(t1_orig)/2), t1_orig))
                    labels_out[2] = labels_out[2].replace('x^', 'sil^')
                    # labels_out = make_concise(labels_out)
                    res = "".join(labels_out)
                    f_after_fixing_syntax.write(res)

