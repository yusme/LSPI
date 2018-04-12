import matplotlib.pyplot as plt

import numpy as np

class Plot:

    def plot_rewad(self,x, y1,y2):

        plt.ylim(-220, 100)

        plt.plot(x, y1,'bo-',linewidth=2.5, linestyle="-", label="LSPI-model-based LSTDQ")
        plt.plot(x, y2,'ro-',linewidth=2.5, linestyle="-", label="LSPI-IS")
        plt.legend(loc='upper left')

        plt.show()





    def plot(self):
        x = np.linspace(0, 30, 30)
        y = np.cos(x / 6 * np.pi) + np.sin(x / 3 * np.pi)

        error = np.random.rand(len(y)) * 2
        y += np.random.normal(0, 0.1, size=y.shape)
        print np.random.normal(0, 0.1, size=y.shape)
        print "\n", np.random.rand(len(y)) * 2
        plt.plot(x, y, 'k', color='#CC4F1B')  # color='#3F7F4C')color="#4682b4"
        plt.fill_between(x, y - error, y + error,
                        edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=1,
                        )
        plt.show()

    def plot_function(self,x,y,z,rbf):
        # plot original data
        plt.figure(figsize=(12, 8))
        plt.plot(x, y, 'k-')

        # plot learned model
        plt.plot(x, z, 'r-', linewidth=2)

        # plot rbfs
        #plt.plot(rbf.centers, np.zeros(rbf.numCenters), 'gs')

        for c in rbf.centers:
            # RF prediction lines
            cx = np.arange(c - 0.7, c + 0.7, 0.01)
            cy = [rbf._basisfunc(np.array([cx_]), np.array([c])) for cx_ in cx]
            # print "-----",cx.shape,len(cy)," "

            #plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

        # print "\n",cx, cy
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
        plt.xlim(-1.2, 1.2)
        # print "plottt"
        plt.show()
