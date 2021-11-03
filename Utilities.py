import csv
class Log:
    def __init__(self, data, folder, file_name, configuration,pNo):
        file_name = folder + '/' + file_name + '-' + '-'.join(map(str, configuration)) + '-' + str(pNo) +'.csv'
        with open(file_name, 'a') as f:
            write = csv.writer(f)
            write.writerows(data)
            
