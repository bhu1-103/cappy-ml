import os
import shutil
import shutil as sh
import pandas as pd


# Input_node_files Location 
input1 = r"E:/kappi/Input/Inputs/"


# Output_simulator Location
output = r"E:/kappi/Output/Outputs/"




var = [
    "sce3a",
    "sce3b",
    "sce3c",
    "sce3d",
    "sce4a",
    "sce4b",
    "sce4c",
    "sce4d",
    "sce5a",
    "sce5b",
    "sce5c",
    "sce5d",
    "sce6a",
    "sce6b",
    "sce6c",
    "sce6d",
    "sce7a",
    "sce7b",
    "sce7c",
    "sce7d",
    "sce8a",
    "sce8b",
    "sce8c",
    "sce8d",
    "sce9a",
    "sce9b",
    "sce9c",
    "sce9d",
    "sce10a",
    "sce10b",
    "sce10c",
    "sce10d",
    "sce11a",
    "sce11b",
    "sce11c",
    "sce11d",
    "sce12a",
    "sce12b",
    "sce12c",
    "sce12d",
    "sce13a",
    "sce13b",
    "sce13c",
    "sce13d",
    "sce14a",
    "sce14b",
    "sce14c",
    "sce14d",
    "sce15a",
    "sce15b",
    "sce15c",
    "sce15d",
    "sce16a",
    "sce16b",
    "sce16c",
    "sce16d",
    "sce17a",
    "sce17b",
    "sce17c",
    "sce17d",
    "sce18a",
    "sce18b",
    "sce18c",
    "sce18d",
    "sce19a",
    "sce19b",
    "sce19c",
    "sce19d",
    "sce20a",
    "sce20b",
    "sce20c",
    "sce20d",
]





#"C:\Users\Koushik P R\Desktop\tosent\"

def foldergen(path, toggle):
    if toggle == "input":
        for i in var:
            os.makedirs(path + "/" + i)
    elif toggle == "output":
        for i in var:
            os.makedirs(path + "/" + i + "_output")


def inputtransfer(trsrc, trdest, tedest):
    for i in var:

        trsrc1 = trsrc + i + "/"

        trdest1 = trdest + i + "/"

        tedest1 = tedest + i + "/"

        print(trsrc1, trdest1, tedest1)

        files = os.listdir(trsrc1)

        # train+test=100 and train = train+validation
        # 50:15:35 split
        train = 65
        print("done")
        for i in range(0, train):
            shutil.copy(trsrc1 + "/" + files[i], trdest1)
        for i in range(train, 100):
            shutil.copy(trsrc1 + "/" + files[i], tedest1)


def outputtransfer(src, dest1, dest2):
    trsrc = ""
    trdest = ""
    tedest = ""
    for p in var:

        airtime = []

        inter = []

        rssi = []

        sinr = []

        tp = []

        trsrc = src + p + "_output/"

        trdest = dest1 + p + "_output/"

        tedest = dest2 + p + "_output/"

        files = os.listdir(trsrc)

        for i in files:
            if i.startswith('airtime'):
                airtime.append(i)
            if i.startswith('interference'):
                inter.append(i)
            if i.startswith('rssi'):
                rssi.append(i)
            if i.startswith('sinr'):
                sinr.append(i)
            if i.startswith('throughput'):
                tp.append(i)

        for ij in range(1, 66):
            shutil.copy(trsrc + "/" + "airtime_" + str(ij) + ".csv", trdest)
            shutil.copy(trsrc + "/" + "rssi_" + str(ij) + ".csv", trdest)
            shutil.copy(trsrc + "/" + "throughput_" + str(ij) + ".csv", trdest)
            shutil.copy(trsrc + "/" + "sinr_" + str(ij) + ".csv", trdest)
            shutil.copy(trsrc + "/" + "interference_" + str(ij) + ".csv", trdest)

        for ik in range(66, 101):
            shutil.copy(trsrc + "/" + "airtime_" + str(ik) + ".csv", tedest)
            shutil.copy(trsrc + "/" + "rssi_" + str(ik) + ".csv", tedest)
            shutil.copy(trsrc + "/" + "throughput_" + str(ik) + ".csv", tedest)
            shutil.copy(trsrc + "/" + "sinr_" + str(ik) + ".csv", tedest)
            shutil.copy(trsrc + "/" + "interference_" + str(ik) + ".csv", tedest)


root_dir = r"E:/kappiinput/"
#"C:\Users\Koushik P R\Desktop\tosent\"
folder_name = input("Enter Folder name: ")
root_dir = root_dir + folder_name

print("Creating Folders")
os.makedirs(root_dir + "/train")
os.makedirs(root_dir + "/test")

os.makedirs(root_dir + "/train/input_node_files")
os.makedirs(root_dir + "/test/input_node_files")
os.makedirs(root_dir + "/train/output_simulator")
os.makedirs(root_dir + "/test/output_simulator")

foldergen(root_dir + "/train/input_node_files", "input")
foldergen(root_dir + "/test/input_node_files", "input")
foldergen(root_dir + "/train/output_simulator", "output")
foldergen(root_dir + "/test/output_simulator", "output")

print("Successfully created Folders")

print("Transferring Data")


print("Transferring Input files")

inputtransfer(input1, root_dir + "/train/input_node_files/", root_dir + "/test/input_node_files/")

print("Success")

print("Transferring Output files")

outputtransfer(output, root_dir + "/train/output_simulator/", root_dir + "/test/output_simulator/")

print("Completed")

print("Successfully Transfered Data")

print("Generating Split files")

j = 0

xx = pd.DataFrame()
yy = pd.DataFrame()

for j in range(0, len(var)):
    for i in range(0, 50):
        xx = xx._append(pd.Series([var[j], i]), ignore_index=True)
        print(var[j] + "," + str(i))
    for i in range(50, 65):
        yy = yy._append(pd.Series([var[j], i]), ignore_index=True)

xx.to_csv(root_dir + "/train/train_split.csv")

yy.to_csv(root_dir + "/train/valid_split.csv")


df1 = pd.read_csv(root_dir + "/train/train_split.csv")
df2 = pd.read_csv(root_dir + "/train/valid_split.csv")

ds = df1.sample(frac=1)
ds2 = df2.sample(frac=1)

ds.to_csv(root_dir+"/train/valid_split_mixed.csv")
ds2.to_csv(root_dir+"/train/train_split_mixed.csv")

print("Success")