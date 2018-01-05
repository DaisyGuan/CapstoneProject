import json
import numpy as np
from collections import OrderedDict
import collections
from ast import literal_eval
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

final_train_result, final_test_result = [], []

def main():
	cryptolen, zeuslen, shadowbrokerslen, lockerlen = 0, 0, 0, 0
	Cryptolist = []
	Lockerslist = []
	Zeuslist = []
	Shadowbrockerlist = []
	length = []
	Malsig_lists = []
	final_result = []
	score = []
	y_cluster = []
	y_train, y_test = [], []
	with open('result.json') as data:
		json_data = json.load(data)
		result_data = json_data
		for event in json_data:
			#print event
			signatures = []
			for call in event['properties']:
				keys = call.keys()
				for key in keys:
					signatures.append(key)
			event['signatures'] = signatures
		for event in json_data:
			if event['label'] == 'Crypto':
				cryptolen += 1
			elif event['label'] == 'Zeus':
				zeuslen +=1
			elif event['label'] == 'shadowbrokers':
				shadowbrokerslen += 1
			elif event['label'] == 'Locker':
				lockerlen += 1
			for signature in event['signatures']:
				Malsig_lists.append(signature)
		system_calls = list(set(Malsig_lists))
		system_calls = [item.encode('utf-8') for item in system_calls]

		#print len(json_data)
		#print cryptolen, zeuslen, shadowbrokerslen, lockerlen
		length = [i for i in range(cryptolen+zeuslen+shadowbrokerslen+lockerlen)]
		shuffle(length)

		for j in length:
			event = result_data[j]
			result_sig, result_array = [], []
			result_sig_cluster, result_array_cluster = [], []
			index = result_data.index(event)
			for call in event['properties']:
				keys = call.keys()
				for key in keys:
					result_sig_cluster.append(key)
			result_sig_cluster = [item.encode('utf-8') for item in result_sig_cluster]
			for call in system_calls:
				if call in result_sig_cluster:
					result_array_cluster.append(1)
				else:
					result_array_cluster.append(0)
			final_result.append(result_array_cluster)
			if event['label'] == 'Crypto':
				y_cluster.append(0)
			elif event['label'] == 'Zeus':
				y_cluster.append(1)
			elif event['label'] == 'shadowbrokers':
				y_cluster.append(2)
			elif event['label'] == 'Locker':
				y_cluster.append(3)

        for



				#y_test.append(event['label'])
				#y_test = [item.encode('utf-8') for item in y_test]
		colormap = np.array(['purple', 'yellow', 'green', 'blue'])
		#print colormap
		#print colormap[y_train]
		maximum = 0
		max_index = 0
		#for i in range(142):
		###
		#pca_cluster = PCA(n_components = 8)
		#cluster_data = pca_cluster.fit_transform(np.asarray(final_result))
		scaler = StandardScaler()
		cluster_data = scaler.fit_transform(final_result)
		kmeans = KMeans(4, random_state = 0)
		model = kmeans.fit(cluster_data)
		score = accuracy_score(y_cluster,kmeans.predict(cluster_data))
			#if score>maximum:
			#	maximum = score
			#	max_index = i
		print score
		#print accuracy_score(y_cluster,kmeans.predict(cluster_data))
		#plt.subplot(1,2,1)
		#plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colormap[y_cluster],s = 5, cmap='viridis')
		#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
		#plt.subplot(1,2,2)
		#plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colormap[model.labels_],s = 5, cmap='viridis')
		#plt.show()
		###
		pca = PCA(n_components=8)
		#pca_test = PCA(n_components=8)
		newData = pca.fit_transform(np.asarray(final_result))
		scaler = StandardScaler()
		newData = scaler.fit_transform(newData)
		train, test, train_labels, test_labels = train_test_split(newData, y_cluster, test_size = 0.3)
		gnb = GaussianNB()
		model_classification = gnb.fit(train, train_labels)
		preds = gnb.predict(test)
		print accuracy_score(test_labels, preds)
		plt.subplot(1,2,1)
		plt.scatter(test[:, 0], test[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
		plt.subplot(1,2,2)
		plt.scatter(test[:, 0], test[:, 1], c=colormap[preds],s = 5, cmap='viridis')
		plt.show()
		#kmeans = KMeans(4, random_state = 0)
		#model = kmeans.fit(trainData)
		#print model.labels_
		#predY = np.choose(model.labels_, [])
		#y_kmeans = kmeans.predict(trainData)
		#score.append(accuracy_score(y_train,kmeans.predict(trainData)))
		#print score.index(max(score))
		#print score[score.index(max(score))]
		#get cluster centers
		#centers=kmeans.cluster_centers_
		#testData = pca_test.fit_transform(np.asarray(final_test_result))
		#testData = scaler.fit_transform(testData)
		#testData = kmeans.predict(testData)
		#plt.subplot(1,2,1)
		#plt.scatter(trainData[:, 0], trainData[:, 1], c=colormap[y_train],s = 5, cmap='viridis')
		#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
		#plt.subplot(1,2,2)
		#plt.scatter(trainData[:, 0], trainData[:, 1], c=colormap[model.labels_],s = 5, cmap='viridis')
		#plt.show()
		#score = accuracy_score(y_test,kmeans.predict(testData))
		#print('Accuracy:{0:f}'.format(score))

	data.close()




if __name__ == '__main__':
    main()
