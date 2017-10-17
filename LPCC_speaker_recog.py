import numpy as np
from sklearn import mixture
from sklearn.externals import joblib
from pyAudioAnalysis import audioBasicIO
import glob
import os
from shutil import copyfile
from scipy.signal import lfilter
from scikits.talkbox import lpc
import math

def extract_LPCCs(file_name):
	features=[]
	for line in open(file_name):
		row=map(float,line.strip().split(' ')) 
		row=row[1:]
		features.append(row)
	return np.array(features)

if __name__=="__main__":

	n_mixtures = 32
	max_iterations = 1000

	speakers={}
	spct=0
	total_sp=len(glob.glob('train_lpccdata/*'))

	if len(glob.glob('train_models/*'))>0:
		for f in glob.glob('train_models/*'):
			os.remove(f)

	for speaker in sorted(glob.glob('train_lpccdata/*')):
		
		print (spct/float(total_sp))*100.0,'% completed'
		speaker_name=speaker.replace('train_lpccdata/','')
		speakers.update({speaker_name:spct})
		
		all_speaker_data=[]

		for sample_file in glob.glob(speaker+'/*.wav_lpcc'):
			features = extract_LPCCs(sample_file)
			if len(all_speaker_data)==0:	all_speaker_data=features
			else:							all_speaker_data=np.concatenate((all_speaker_data,features),axis=0)

		print all_speaker_data.shape

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
			tct=len(glob.glob('test_lpccdata/'+speaker+'/*.wav_lpcc'))
		for testcasefile in glob.glob('test_lpccdata/'+speaker+'/*.wav_lpcc'):
			features = extract_LPCCs(testcasefile)
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





