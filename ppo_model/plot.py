import matplotlib.pyplot as plt

f = open("20220411.txt", "r")

loss_list = []

for line in f.readlines():
    if "loss" in line and "_" not in line:
        loss = line.split("|")
        # print(float(loss[2].strip()))
        loss_list.append(float(loss[2].strip()))

# print(loss_list)

plt.title("Loss curve for the Robot Grasping Task in DR Setting")
plt.xlabel("episodes")
plt.ylabel("loss")
plt.plot(list(range(len(loss_list))), loss_list)
plt.show()