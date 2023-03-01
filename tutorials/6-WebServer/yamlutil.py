import ruamel.yaml as ruaml
import os

yaml = ruaml.YAML()

# 获取yaml文件路径
yamlPath = os.path.join("./", "config.yml")


class YamlUtil:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            print("YamlUtil first init")
            cls.__instance = super(YamlUtil, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def read(self, k):
        with open(yamlPath, encoding="utf-8") as f:
            x = yaml.load(f)
            if k in x:
                value = x[k]
                return value
            else:
                return ''

    def write(self, k, v):
        with open(yamlPath, "r", encoding="utf-8") as f:
            x = yaml.load(f)
            with open(yamlPath, "w", encoding="utf-8") as f:
                x[k] = v
                return yaml.dump(x, f)
        # return "error"

    def write2(self, k0, k1, v): #default k1 is None
        with open(yamlPath, "r", encoding="utf-8") as f:
            x = yaml.load(f)

            if k1 == None:
                with open(yamlPath, "w", encoding="utf-8") as f:
                    x[k0] = v
                    return yaml.dump(x, f)
            elif k0 in x:
                with open(yamlPath, "w", encoding="utf-8") as f:
                    x[k0][k1] = v
                    return yaml.dump(x, f)

                # if k1 in x[k0]:
                #     with open(yamlPath, "w", encoding="utf-8") as f:
                #         x[k0][k1] = v
                #         return yaml.dump(x, f)
        # return "error"

    def delete(self, k):
        with open(yamlPath, "r", encoding="utf-8") as f:
            x = yaml.load(f)
            if k in x:
                with open(yamlPath, "w", encoding="utf-8") as f:
                    del x[k]
                    return yaml.dump(x, f)

        # return "error"

# yamlUtil = YamlUtil()
