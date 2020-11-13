from natsort import natsorted
import os
import glob
class FilePath():
    def __init__(self, image_folder, csv_folder):
        self.image_folder = image_folder
        self.csv_folder = csv_folder
        self.image_path = os.path.join("../image-data/",self.image_folder)
        self.csv_path = os.path.join("csv/label_name_data",self.csv_folder,self.csv_folder + ".csv")

    def image(self):
        return self.image_path

    def csv(self):
        return self.csv_path

    def for_mix_image(self):
        self.classify_image_path = natsorted(
                glob.glob(os.path.join("../image-data",self.image_folder,"classify/*")))
        return self.classify_image_path

    def for_mix_csv(self):
        self.classify_csv_path = natsorted(
                glob.glob(os.path.join("csv/label_name_data",self.csv_folder + "/*")))
        return self.classify_csv_path

    def for_train_shadow_image(self,num):
        self.train_shadow_image_path = os.path.join(
                "../image-data",self.image_folder + "_bright_about" + num)
        return self.train_shadow_image_path

    def for_train_shadow_csv(self,num):
        self.train_shadow_csv_path = os.path.join(
                "csv/label_name_data/train_shadow" ,self.csv_folder + "_bright_about" + num + ".csv")
        return self.train_shadow_csv_path 

    def for_param_csv(self):
        self.param_csv_path = os.path.join("csv/analysis",self.csv_folder + "_param.csv")
        return self.param_csv_path
