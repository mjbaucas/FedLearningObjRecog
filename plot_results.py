from matplotlib import pyplot as plt

dataX = [1,2,3,4,5,6,7,8,9,10]
dataLY = [93.72, 92.88, 91.18, 91.62, 91.18, 91.40, 91.68, 91.59, 91.60, 91.50]
dataGY = [93.72, 93.81, 93.57, 93.57, 93.57, 93.62, 93.62, 93.62, 93.72, 93.72]


def plot_data(dataX, dataLY, dataGY):
    plt.figure(0)
    plt.plot(dataX,dataGY,marker='o',label="Global")
    plt.plot(dataX,dataLY,marker='s',label="Local (Averaged)")
    plt.xlabel("Number of Clients")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='center right')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.savefig('accmodel.png')

plot_data(dataX, dataLY, dataGY)