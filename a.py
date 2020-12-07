import cv2
import pandas as pd

#img = cv2.imread("../image-data/original/H_1_1s.JPG")
#cv2.imshow("img",img)
#cv2.waitKey()

write_csvpath = "pred_result/test_11_6_retry_result_noif.csv"

df = pd.read_csv(write_csvpath,usecols=[1])
df_bool_normal = (df == "normal pipe image")
df_bool_abnormal = (df == "abnormal pipe image")

a = df_bool_normal.sum().values
b = df_bool_abnormal.sum().values
total = a+b
per = (b/total)*100
write = pd.DataFrame(data = [a,b,total,per])
write = write.T
print(write)

header = ["normal","abnormal","total","ab_per"]
write.to_csv(write_csvpath,mode = "a" ,index =False,header = header)
