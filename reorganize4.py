import json
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

final_train_result, final_test_result = [], []



def main():
	cryptolen, zeuslen, shadowbrokerslen, lockerlen = 0, 0, 0, 0
	length = []
	Malsig_lists = []
	final_result = []
	y_labels = []
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
			#index = result_data.index(event)
			for call in event['properties']:
				keys = call.keys()
				for key in keys:
					result_sig.append(key)
			result_sig = [item.encode('utf-8') for item in result_sig]
			for call in system_calls:
				if call in result_sig:
					result_array.append(1)
				else:
					result_array.append(0)
			final_result.append(result_array)
			if event['label'] == 'Crypto':
				y_labels.append(0)
			elif event['label'] == 'Zeus':
				y_labels.append(1)
			elif event['label'] == 'shadowbrokers':
				y_labels.append(2)
			elif event['label'] == 'Locker':
				y_labels.append(3)

		colormap = np.array(['purple', 'yellow', 'green', 'blue'])

		#---------Using Different Classification Algorithm---------------------#
		train, test, train_labels, test_labels = train_test_split(final_result, y_labels, test_size = 0.15)
		pca = PCA(n_components=8)
		test_draw = pca.fit_transform(np.asarray(test))
		scaler = StandardScaler()
		test_draw = scaler.fit_transform(test_draw)


		#------1. Gaussian Processes---------#
		gnb = GaussianNB()
		model_classification = gnb.fit(train, train_labels)
		preds_gnb = gnb.predict(test)
		print "Gaussian: "
		print accuracy_score(test_labels, preds_gnb)

		plt.suptitle("Gaussian")
		plt.subplot(1,2,1)
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		plt.subplot(1,2,2)
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_gnb],s = 5, cmap='viridis')
		#plt.show()

		#------2. Random Forest--------#
		random_forest = RandomForestClassifier(n_jobs=-1, random_state=0, max_features = None)
		random_forest.fit(train, train_labels)
		preds_random_forest = random_forest.predict(test)
		print "Random Forest: "
		print accuracy_score(test_labels, preds_random_forest)

		plt.suptitle("Random Forest")
		plt.subplot(1,2,1)
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		plt.subplot(1,2,2)
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_random_forest],s = 5, cmap='viridis')
		#plt.show()

		#------3. Neural Network------#
		#neural_network = MLPClassifier(solver = "lbfgs", alpha = 1e-6,
		#								hidden_layer_sizes = (10,13), random_state = 1)
		#neural_network = MLPClassifier(solver = "adam", alpha = 1e-6,
		#								hidden_layer_sizes = (10,13), random_state = 1)
		#neural_network = MLPClassifier(solver = "lbfgs", alpha = 1e-4,
		#								hidden_layer_sizes = (10,13), random_state = 1)
		#neural_network = MLPClassifier(solver = "lbfgs", alpha = 1e-5,
		#								hidden_layer_sizes = (10,13), random_state = 1)
		neural_network_1 = MLPClassifier(solver = "lbfgs", alpha = 1e-6,
										hidden_layer_sizes = (10,13), random_state = 1)
		#neural_network_2 = MLPClassifier(solver = "sgd", alpha = 1e-6,
		#								hidden_layer_sizes = (10,13), random_state = 1)
		#neural_network_2 = MLPClassifier(solver = "lbfgs", alpha = 1e-6,
		#								hidden_layer_sizes = (10,18), random_state = 1)
		#neural_network.fit(train, train_labels)
		neural_network_1.fit(train, train_labels)
		#neural_network_2.fit(train, train_labels)
		#preds_neural_network = neural_network.predict(test)
		preds_neural_network_1 = neural_network_1.predict(test)
		#preds_neural_network_2 = neural_network_2.predict(test)
		print "Neural Network: "
		#print accuracy_score(test_labels, preds_neural_network)
		print accuracy_score(test_labels, preds_neural_network_1)
		#print accuracy_score(test_labels, preds_neural_network_2)

		plt.suptitle("Neural Network")
		plt.subplot(2,2,1)
		plt.subplot(2,2,1).set_title('a')
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		plt.subplot(2,2,2)
		plt.subplot(2,2,2).set_title('b')
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_neural_network],s = 5, cmap='viridis')
		plt.subplot(2,2,3)
		plt.subplot(2,2,3).set_title('c')
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_neural_network_1],s = 5, cmap='viridis')
		plt.subplot(2,2,4)
		plt.subplot(2,2,4).set_title('d')
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_neural_network_2],s = 5, cmap='viridis')
		plt.show()

		#------4. SVM(polynomial degree 3 kernel)--------#
		SVM = svm.SVC(kernel = 'poly', degree = 3, C = 1.0)
		SVM.fit(train, train_labels)
		preds_SVM = SVM.predict(test)
		print "SVM: "
		print accuracy_score(test_labels, preds_SVM)
		#plots[3, 0].set_title("SVM")
		#plots[3, 0].scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		#plots[3, 1].scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_SVM],s = 5, cmap='viridis')
		#plt.suptitle("SVM")
		#plt.subplot(5,2,7)
		#plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		#plt.subplot(5,2,8)
		#plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_SVM],s = 5, cmap='viridis')

		#------5. KNN --------#
		KNN = KNeighborsClassifier(n_neighbors = 30, weights = "distance")
		KNN.fit(train, train_labels)
		preds_KNN = KNN.predict(test)
		print "KNN: "
		print accuracy_score(test_labels, preds_KNN)
		plt.suptitle("KNN")
		plt.subplot(1,2,1)
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[test_labels],s = 5, cmap='viridis')
		plt.subplot(1,2,2)
		plt.scatter(test_draw[:, 0], test_draw[:, 1], c=colormap[preds_KNN],s = 5, cmap='viridis')
		plt.show()



	data.close()




if __name__ == '__main__':
    main()
