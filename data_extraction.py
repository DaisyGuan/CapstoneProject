import json
from collections import OrderedDict

def main():
    
    with open('dataset.json') as json_data:
        d = json.load(json_data)
        result_final = []
        for data in d:
            result_dic = OrderedDict()
            result_dic['id'] = data["_id"]
            result_dic['label'] = None
            result_dic['properties'] = []
            for key, value in data['properties'].items():
                if key == "label":
                    result_dic['label'] = data['properties'][key]
                    #print data['properties'][key]
                else:
                    data['properties'][key] = "True"
                    dic = {'action':'False'}
                    dic[key] = dic.pop('action')
                    dic[key] = data['properties'][key]
                    result_dic['properties'].append(dic)
                    #result_dic['properties'].append(dic)
            #print result_dic
            result_dic = json.dumps(result_dic)
            #print result_dic
            result_final.append(result_dic)
        print type(result_final)
    json_data.close()
    output = open('result_Lockers.json','w+')
    for item in result_final:
        output.write("%s," % item)
    output.close()

    


if __name__ == '__main__':
    main()