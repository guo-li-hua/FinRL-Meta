from yamlutil import YamlUtil

cfg = YamlUtil()


# data
def curent_data_source_get():
    return cfg.read('curent_data_source')


def indicators_get():
    return cfg.read('indicators')


def factors_get():
    facts = cfg.read('factors')
    if "fft" in facts:
        facts += ["absolute0", "absolute1", "absolute2", "absolute3", "absolute4", "absolute5", "angle0", "angle1",
                  "angle2", "angle3", "angle4", "angle5"]

        def filter_func(item):
            if "fft" in item:
                return False
            else:
                return True

        facts_list = filter(filter_func, facts)
        print(list(facts_list))
        return list(facts_list)
    return facts

def factors_cfg_get():
    return cfg.read('factors')


# dir
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
