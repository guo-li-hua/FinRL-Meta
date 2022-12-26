from yamlutil import YamlUtil

cfg = YamlUtil()


#data
def curent_data_source_get():
    return cfg.read('curent_data_source')

def indicators_get():
    return cfg.read('indicators')

def factors_get():
    return cfg.read('factors')

#dir
def dir_list_get(dir_name=None):
    if dir_name is None:
        return cfg.read('dir_list')
    else:
        return cfg.read('dir_list')[dir_name]

def dir_list_set(k, v):
    cfg.write(k, v)

#
def ticker_list_get():
    return cfg.read('ticker_list')

def ticker_list_add(ticker):
    return cfg.write2('ticker_list', None, ticker)

def ticker_list_del(ticker):
    return cfg.delete(ticker)

#
def time_list_get():
    return cfg.read("time_list")

def time_list_set(k, v):
    return cfg.write2("time_list", k, v)

def time_list_del(k):
    return cfg.delete(k)
