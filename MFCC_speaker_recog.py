import numpy as np
from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE
from sklearn import mixture
from sklearn.externals import joblib
import glob
import os
from shutil import copyfile
from scipy.fftpack import fft,ifft

def extract_MFCCs(x,Fs,window_size,overlap,VTH_Multiplier=0.05,VTH_range=100,deltas=False):
	
	energy = [ s**2 for s in x]
	Voiced_Threshold = VTH_Multiplier*np.mean(energy)
	clean_samples=[0]
	
	for sample_set in xrange(0,len(x)-VTH_range,VTH_range):
		sample_set_th = np.mean(energy[sample_set:sample_set+VTH_range])
		if sample_set_th>Voiced_Threshold:
			clean_samples.extend(x[sample_set:sample_set+VTH_range])

	energy=None

	try:
		MFCC13_F = AFE.stFeatureExtraction(clean_samples, Fs, window_size, overlap)
	except:
		print "ERROR:extract_MFCCs: Error while trying to extract mfccs"
		return []
	
	clean_samples=None

	if not deltas:
		return MFCC13_F
	
	delta_MFCC = np.zeros(MFCC13_F.shape)
	for t in range(delta_MFCC.shape[1]):
		index_t_minus_one,index_t_plus_one=t-1,t+1
		
		if index_t_minus_one<0:    
			index_t_minus_one=0
		if index_t_plus_one>=delta_MFCC.shape[1]:
			index_t_plus_one=delta_MFCC.shape[1]-1
		
		delta_MFCC[:,t]=0.5*(MFCC13_F[:,index_t_plus_one]-MFCC13_F[:,index_t_minus_one] )
	
	
	double_delta_MFCC = np.zeros(MFCC13_F.shape)
	for t in range(double_delta_MFCC.shape[1]):
		
		index_t_minus_one,index_t_plus_one, index_t_plus_two,index_t_minus_two=t-1,t+1,t+2,t-2
		
		if index_t_minus_one<0:
			index_t_minus_one=0
		if index_t_plus_one>=delta_MFCC.shape[1]:
			index_t_plus_one=delta_MFCC.shape[1]-1
		if index_t_minus_two<0:
			index_t_minus_two=0
		if index_t_plus_two>=delta_MFCC.shape[1]:
			index_t_plus_two=delta_MFCC.shape[1]-1
		  
		double_delta_MFCC[:,t]=0.1*( 2*MFCC13_F[:,index_t_plus_two]+MFCC13_F[:,index_t_plus_one]
									-MFCC13_F[:,index_t_minus_one]-2*MFCC13_F[:,index_t_minus_two] )
	
	Combined_MFCC_F = np.concatenate((MFCC13_F,delta_MFCC,double_delta_MFCC),axis=1)
	
	return Combined_MFCC_F


if __name__=="__main__":

	window = 0.020
	window_overlap = 0.010
	voiced_threshold_mul = 0.05
	voiced_threshold_range=100
	n_mixtures = 128
	max_iterations = 75
	calc_deltas=True
	
	# for tf in glob.glob('train_wavdata/*'):
	# 	name=tf.replace("train_wavdata/","")
	# 	copyfile(tf+"/7.wav", "test_wavdata/"+name+"/7.wav")
	# 	os.remove(tf+"/7.wav")
	# 	copyfile(tf+"/8.wav", "test_wavdata/"+name+"/8.wav")
	# 	os.remove(tf+"/8.wav")


	speakers={}
	spct=0
	total_sp=len(glob.glob('train_wavdata/*'))

	if len(glob.glob('train_models/*'))>0:
		for f in glob.glob('train_models/*'):
			os.remove(f)

	for speaker in sorted(glob.glob('train_wavdata/*')):
		
		print (spct/float(total_sp))*100.0,'% completed'
		speaker_name=speaker.replace('train_wavdata/','')
		speakers.update({speaker_name:spct})
		
		all_speaker_Fs,all_speaker_data=0,[]

		for sample_file in glob.glob(speaker+'/*.wav'):

			[Fs, x] = audioBasicIO.readAudioFile(sample_file)
			
			if all_speaker_Fs==0:	all_speaker_Fs=Fs

			if Fs==all_speaker_Fs:
				features = extract_MFCCs(x,Fs,window*Fs,window_overlap*Fs,voiced_threshold_mul,voiced_threshold_range,calc_deltas)
				if len(all_speaker_data)==0:	all_speaker_data=features
				else:							all_speaker_data=np.concatenate((all_speaker_data,features),axis=0)
			else:	print sample_file+" skipped due to mismatch in frame rate"

		try:
			gmm = mixture.GaussianMixture(n_components=n_mixtures, covariance_type='diag' , max_iter = max_iterations ).fit(all_speaker_data)
		except:
			print "ERROR : Error while training model for file "+speaker
		
		try:
			joblib.dump(gmm,'train_models/'+speaker_name+'.pkl')
		except:
			 print "ERROR : Error while saving model for "+speaker_name

		spct+=1

	print "Training Completed"

	confusion_matrix = np.zeros((total_sp,total_sp))
	tct=0
	for speaker in speakers:
		if tct<=0:
			tct=len(glob.glob('test_wavdata/'+speaker+'/*.wav'))
		for testcasefile in glob.glob('test_wavdata/'+speaker+'/*.wav'):
			[Fs, x] = audioBasicIO.readAudioFile(testcasefile)
			features = extract_MFCCs(x,Fs,window*Fs,window_overlap*Fs,voiced_threshold_mul,voiced_threshold_range,calc_deltas)
			max_score=-9999999
			max_speaker=speaker
			for modelfile in sorted(glob.glob('train_models/*.pkl')):
				gmm = joblib.load(modelfile) 
				score=gmm.score(features)
				if score>max_score:
					max_score,max_speaker=score,modelfile.replace('train_models/','').replace('.pkl','')
			print speaker+" -> "+max_speaker+(" Y" if speaker==max_speaker  else " N")
			confusion_matrix[ speakers[speaker] ][speakers[max_speaker]]+=1

	print "Accuracy: ",(sum([ confusion_matrix[i][j] if i==j  else 0 for i in xrange(total_sp) for j in xrange(total_sp) ] )*100)/float(tct*total_sp)





