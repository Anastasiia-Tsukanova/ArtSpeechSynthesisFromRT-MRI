import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from os import walk, listdir, makedirs
from os.path import isfile, exists
from pydub import AudioSegment
from os.path import join as merge_path
import sys

MERGED_PARS = False
CALCULATE_LIPS = True
CALCULATE_VELUM = True
PRINT_INIT_PAU = False
PRINT_END_SIL = False

def ensure_dir(dir_path):
	if not exists(dir_path):
		makedirs(dir_path)

def read_binfile(filename, dim=60, dtype=np.float32):
    #Reads binary file into numpy array.
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    return v_data

non_art_param_extensions_to_save = ['lf0', 'lab']
non_art_param_extensions_to_draw = ['lf0']
lip_dist_extensions = ['ldi'] if not MERGED_PARS else ['ldc']
lip_cont_extensions = ['lcs']
lip_protr_extensions = ['llp', 'lbp']
lips_extensions = lip_dist_extensions + lip_cont_extensions + lip_protr_extensions
vel_dist_extensions = ['vdt', 'vdw', 'vtw']
vel_cont_extensions = ['vct', 'vcw', 'ctw']
velum_extensions = vel_dist_extensions + vel_cont_extensions

articulators = lips_extensions + velum_extensions
# articulators = ['ldc', 'lbp', 'llp']
lips = {'Lips': [('open_or_close', 'lcl'), ('dist_lips', 'ldi'), ('contact_lips', 'lcs'), ('protrusion_bottom_lip', 'lbp'), ('protrusion_upper_lip', 'llp')]}
# lips = {'Lips': [('dist_cont_lips', 'ldc'), ('protrusion_bottom_lip', 'lbp'), ('protrusion_upper_lip', 'llp')]}
velum = {'Velum': [('dists_vel_tong', 'vdt'), ('dists_vel_wall', 'vdw'), ('dists_tong_wall', 'vtw'), ('contacts_vel_tong', 'vct'), ('contacts_vel_wall', 'vcw'), ('contacts_tong_wall', 'ctw')]}
rename_art = {'lcl': 'ls_op_cl', 'ldi': 'ls_dist', 'lcs': 'ls_cont', 'lbp': 'lw_l_pr', 'llp': 'up_l_pr',
			  'vdt': 'v_t_dist', 'vdw': 'v_w_dist', 'vtw': 't_w_dist', 'vct': 'v_t_cont', 'vcw': 'v_w_cont', 'ctw': 't_w_cont',
			  'lf0': 'f0'}
full_art_name = dict()
for art, art_params in {**lips, **velum}.items():
	for long_n, short_n in art_params:
		# print('%s --> %s' % (short_n, long_n))
		full_art_name[short_n] = long_n
sampa2merlin_loc = {'||': '||', 'a': 'a', 'l': 'l', 'O': 'oopen', 'R': 'r', 'sil': 'sil', 'pau': 'pau', 'e': 'e', 'z': 'z', 'v': 'v', 'a~': 'an', 't': 't', 'Z': 'zh', 'e~': 'en', 'k': 'k', 'o~': 'on', 'n': 'n', 'j': 'j', 'J': 'gn', 'd': 'd', 'b': 'b', 'i': 'i', 'p': 'p', 'm': 'm', 'E': 'eps', '@': 'schwa', 'y': 'y', 'o': 'o', 'u': 'u', 's': 's', '2': 'deux', 'S': 'sh', 'f': 'f', 'w': 'w', '9~': 'oen', 'H': 'h', 'g': 'g', '9': 'oe', 'N': 'nn', 'x': 'x'}
represent = {merlin_ph: sampa_ph for sampa_ph, merlin_ph in sampa2merlin_loc.items()}

def visualize(gen_dir, id_file, wav_dir, lab_dir, feat_dir, fig_dir, used_params_nb_dir=None, rate=5):
	with open(id_file) as f_w_list_of_files:
		synthesis_results = [rec.replace('\n', '').replace('\r', '') for rec in f_w_list_of_files.readlines()]
	
	for synth_res in synthesis_results:
		if synth_res == '':
			continue
		to_plot = dict()
		lab_f = merge_path(lab_dir, synth_res+'.lab')
		phs = list()
		last_t_treated = -rate
		time_tick_indexes = list()
		if isfile(lab_f):
			with open(lab_f) as lab_file:
				full_labels = lab_file.readlines()
				for label in full_labels:
					if len(label) > 2:
						curr_ph = label[label.find('-')+1:label.find('+')]
						label_spl = label.split(' ')
						t_1, t_2 = int(float(label_spl[0])/10000), int(float(label_spl[1])/10000)
						while(last_t_treated <= t_2):
							if len(phs) == 0 or (phs[-1] != '||' and curr_ph != phs[-1]):
								if True in time_tick_indexes:
									last_true_id = time_tick_indexes[::-1].index(True)
									time_tick_indexes[-int(round(last_true_id*0.5))] = True
								time_tick_indexes.append(True)
								phs.append('||')
							else:
								time_tick_indexes.append(False)
								phs.append(curr_ph)
							last_t_treated += rate
				first_accepted_init_pau_id = 0
				last_accepted_end_sil_id = 0
				if not PRINT_INIT_PAU:
					for ph_id, ph in enumerate(phs):
						if ph not in ['pau', 'sil', 'x', '||']:
							break
						if ph_id > 5:
							first_accepted_init_pau_id = ph_id - 5
				if not PRINT_END_SIL:
					for ph_id, ph in enumerate(reversed(phs)):
						if ph not in ['pau', 'sil', 'x', '||']:
							break
						if ph_id > 5:
							last_accepted_end_sil_id = - ph_id + 5
		time = None
		setting_ph_t_up = True
		print("Creating the articulatory parameter plots for %s..." % synth_res)
		for art in articulators+non_art_param_extensions_to_save:
			# print('save ' + art)
			art_f = merge_path(feat_dir, synth_res+'.'+art)
			# if art in ['lab', 'labbin'] and isfile(art_f) and used_params_nb_dir is not None:
			# 	print('Binary %s exists.' % art_f)
			# 	with open(art_f, 'rb') as lab_binary:
			# 		labels_in_use = lab_binary.read()
			# 		print(labels_in_use.decode('utf-8'))
			# 		continue
			if isfile(art_f):
				art_seq = read_binfile(art_f)
				to_plot[art] = art_seq
				if art == 'lf0':
					to_plot[art] = np.exp(to_plot[art])
				if setting_ph_t_up:
					time = np.arange(0, art_seq.size*rate, rate)
					time_tick_indexes = np.array(time_tick_indexes[:art_seq.size])
					phs = np.array(phs[:art_seq.size])
				if used_params_nb_dir is not None:
					ensure_dir(merge_path(used_params_nb_dir, art if art != 'lf0' else 'f0'))
					art_labeled = [('%d'%t).ljust(10) + ' ms: ' + ('['+ph+']').ljust(7) + ' %10.5f'%v for (t, ph, v) in zip(time, phs, to_plot[art])]
					with open(merge_path(used_params_nb_dir, art if art != 'lf0' else 'f0', synth_res+'.'+(art if art != 'lf0' else 'f0')), 'w') as art_l_f:
						art_l_f.write('\n'.join(art_labeled))
				to_plot[art] = to_plot[art][first_accepted_init_pau_id:last_accepted_end_sil_id]
				to_plot[art][np.where(to_plot[art] < 0)] = 0
				if art in vel_cont_extensions:
					to_plot[art][np.where(to_plot[art] > 1)] = 1
				if setting_ph_t_up:
					# print(art)
					# print(first_accepted_init_pau_id)
					phs = phs[first_accepted_init_pau_id:last_accepted_end_sil_id]
					# print(phs, phs.shape)
					time = time[first_accepted_init_pau_id:last_accepted_end_sil_id]
					# print(time, time.shape)
					time_tick_indexes[first_accepted_init_pau_id] = time_tick_indexes[0] 
					time_tick_indexes = time_tick_indexes[first_accepted_init_pau_id:last_accepted_end_sil_id]
					
					# print(time_tick_indexes, time_tick_indexes.shape)
					phs = np.array([represent[ph] for ph in phs])
					setting_ph_t_up = False
			else:
				print("Cannot find %s" % art_f)
				to_plot[art] = None
		wav_f = merge_path(wav_dir, synth_res+'.wav')
		fig_width = 0.005*len(AudioSegment.from_file(wav_f))
		# print(fig_width)
		plt.clf()
		#matplotlib.rc('xtick', labelsize=5)
		plt.rcParams["figure.figsize"]=(fig_width, 10)
		plt.figure(1)
		plt.title(synth_res)
		for art_id, art in enumerate(articulators+non_art_param_extensions_to_draw):
			# print('plot ' + art)
			if art in to_plot and to_plot[art] is not None:
				plt.subplot(len(articulators) + len(non_art_param_extensions_to_draw), 1, art_id+1)
				# ax.annotate('Test', (time[700], to_plot[art][700]), xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
				ax = plt.gca()
				ax.set_xlabel('phonemes')
				ax.set_ylabel(rename_art[art])
				ax.set_title(full_art_name[art] if art in articulators else rename_art[art])
				ax.set_xticks(time[time_tick_indexes])
				ax.set_xticklabels(phs[time_tick_indexes])
				# to_plot[art]print(None in time)
				if art == 'ldc':
					collection = collections.BrokenBarHCollection.span_where(time, ymin=0, ymax=np.amax(to_plot[art]), where=to_plot[art] > 0, facecolor='green', alpha=0.5)
					ax.add_collection(collection)
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=0, where=to_plot[art] < 0, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				if art in lip_dist_extensions+vel_dist_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=np.amax(to_plot[art]), where=to_plot[art] < 1, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				elif art in lip_cont_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=np.amax(to_plot[art]), where=to_plot[art] >= 0.9, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				elif art in vel_cont_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=np.amax(to_plot[art]), where=to_plot[art] >= 0.7, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				elif art in lip_protr_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=0, ymax=np.amax(to_plot[art]), where=to_plot[art] > (14 if art == 'llp' else 14), facecolor='green', alpha=0.5)
					ax.add_collection(collection)
				# print(time, time.shape)
				# print(to_plot[art], to_plot[art].shape)
				# print(art)
				plt.plot(time, to_plot[art])
		ensure_dir(fig_dir)
		plt.tight_layout()
		plt.savefig(merge_path(fig_dir, synth_res+'.png')) # synth_res+'-'+art+'.png'))
		for art_id, art in enumerate(lip_dist_extensions + lip_cont_extensions+velum_extensions+non_art_param_extensions_to_draw):
			if art in to_plot and to_plot[art] is not None:
				plt.clf()
				plt.rcParams["figure.figsize"]=(fig_width, 10)
				plt.figure(1)
				plt.title(synth_res + ' - ' + art)
				ax = plt.gca()
				ax.set_xlabel('phonemes')
				ax.set_ylabel(rename_art[art])
				ax.set_title(full_art_name[art] if art in articulators else rename_art[art])
				ax.set_xticks(time[time_tick_indexes])
				ax.set_xticklabels(phs[time_tick_indexes])
				if art in lip_dist_extensions+vel_dist_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=np.amax(to_plot[art]), where=to_plot[art] < 1, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				elif art in lip_cont_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=np.amax(to_plot[art]), where=to_plot[art] >= 0.9, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				elif art in vel_cont_extensions:
					collection = collections.BrokenBarHCollection.span_where(time, ymin=np.amin(to_plot[art]), ymax=np.amax(to_plot[art]), where=to_plot[art] >= 0.7, facecolor='red', alpha=0.5)
					ax.add_collection(collection)
				# print(art)
				plt.plot(time, to_plot[art])
				ensure_dir(merge_path(fig_dir, synth_res))
				plt.savefig(merge_path(fig_dir, synth_res, synth_res+'-'+art+'.png'))
				# ensure_dir(merge_path(fig_dir, art))
		if all(art in to_plot and to_plot[art] is not None for art in lip_protr_extensions):
			plt.clf()
			plt.figure(1, figsize=(fig_width, 15))
			plt.title(synth_res + ' - ' + '-'.join(lip_protr_extensions))
			for art_id, art in enumerate(lip_protr_extensions):
				plt.subplot(len(lip_protr_extensions), 1, art_id+1)
				# ax.annotate('Test', (time[700], to_plot[art][700]), xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
				ax = plt.gca()
				ax.set_xlabel('phonemes')
				ax.set_ylabel(rename_art[art])
				if art_id == 0:
					ax.set_title('Lip protrusion')
				ax.set_xticks(time[time_tick_indexes])
				ax.set_xticklabels(phs[time_tick_indexes])
				# to_plot[art]print(None in time)
				collection = collections.BrokenBarHCollection.span_where(time, ymin=0, ymax=np.amax(to_plot[art]), where=to_plot[art] > (14 if art == 'llp' else 14), facecolor='green', alpha=0.5)
				ax.add_collection(collection)
				# print(art)
				plt.plot(time, to_plot[art])
			ensure_dir(merge_path(fig_dir, synth_res))
			plt.savefig(merge_path(fig_dir, synth_res, synth_res+'-'+'-'.join(lip_protr_extensions)+'.png'))

if __name__ == '__main__':
	if len(sys.argv)<1:
		print('Usage: python visualize_art_params.py <test_synthesis_dir>\n')
		sys.exit(0)
	gen_dir   = sys.argv[1] #  '/home/anastasiia/Work/merlin/egs/build_your_own_voice/antoine_art_new_labels/experiments/antoine_articulatory_voice/test_synthesis_second_set/'
	id_file = merge_path(gen_dir, 'test_id_list.scp')
	wav_dir = merge_path(gen_dir, 'wav')
	lab_dir = merge_path(gen_dir, 'gen-lab')
	feat_dir = merge_path(gen_dir, 'wav')
	fig_dir = merge_path(gen_dir, 'fig')
	used_params_nb_dir = merge_path(gen_dir, 'used_params_nb')
	visualize(gen_dir, id_file, wav_dir, lab_dir, feat_dir, fig_dir, used_params_nb_dir)
	
	
