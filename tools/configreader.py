import configparser
import os


def absolute_path(filename,code_dir = 'cluster'):
    current_path = os.path.abspath(__file__)
    base_dir = current_path.split(code_dir)[0]
    filepath = os.path.join(str(base_dir)+code_dir,filename)
    return filepath


def reader(relative_path):
    config_path = absolute_path(relative_path)
    try:
        config = configparser.ConfigParser()
        config.read(config_path,encoding='utf8')
        return config
    except FileNotFoundError:
        return None


model_congig_path = 'config/model.ini'

model_config = reader(model_congig_path)


if __name__ == '__main__':
    print(os.path.abspath('.'))
    print(os.getcwd())
    print(os.path.abspath(__file__))