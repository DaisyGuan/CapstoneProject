import json
import numpy as np
from collections import OrderedDict

def main():
	system_calls = []
	result = []
	properties = []
	with open('result.json') as data:
		json_data = json.load(data)
		for event in json_data:
			#print event
			for call in event['properties']:
				keys = call.keys()
				for key in keys:
					if key not in system_calls:
						system_calls.append(key)
		print type(system_calls)
		#system_calls = np.array(str(system_calls).replace('u\'','\''))
		#system_calls = system_calls.tolist()
		#print type(system_calls)
		for event in json_data:
			#print event
			result_dic = OrderedDict()
			result_dic['id'] = event['id']
			result_dic['label'] = event['label']
			for call in event['properties']:
				keys = call.keys()
				for key in keys:
					for each_call in system_calls:
						if key == each_call:
							dic = {}
							dic[key] = "1"
							properties.append(dic)
						else:
							dic = {}
							dic[each_call] = "0"
							properties.append(dic)
			result_dic['properties'] = properties
			result.append(result_dic)
			#print result
	data.close()
	with open('result_reorganize.json','w') as outfile:
		json.dump(result, outfile)
	outfile.close()



if __name__ == '__main__':
    main()

			