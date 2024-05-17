import logging
import shutil

class Misakalog:
    def __init__(self, config):
        # save config file
        logname = "./log/"+"log_of_misaka_"+config.TrainId+'-'+str(config.step)
        try:
            f1 = open("config.py", encoding="utf-8")
            f2 = open(logname, "a", encoding="utf-8")
            f2.write("\n\n********************************** the config of exeperience ********************************** \n\n")
            shutil.copyfileobj(f1, f2)
            f2.write("\n\n*********************************************************************************************** \n\n")
        finally:
            if(f1):
                f1.close()
            if (f2):
                f2.close()


        self.logg = logging.getLogger("log_of_misaka_"+config.TrainId)
        self.logg.handlers = []

        self.FORMATTER = logging.Formatter("%(asctime)s-%(message)s")

        self.p_stream = logging.StreamHandler()

        self.f_stream = logging.FileHandler(logname, mode="a", encoding="utf-8")

        self.p_stream.setFormatter(self.FORMATTER)
        self.f_stream.setFormatter(self.FORMATTER)

        self.logg.addHandler(self.p_stream)
        self.logg.addHandler(self.f_stream)

        self.logg.setLevel(logging.DEBUG)

    def printInfo(self, str):
        self.logg.info(str)





