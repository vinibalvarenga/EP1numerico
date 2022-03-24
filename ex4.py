import matplotlib.pyplot as plt
import numpy as np

#Epsilon
Ep = 10**(-13)

def nomeiaPontos1():
    #lista com os pontos ordenados
    guardaPosicao = [0, 5, 0, 4, 0, 3, 0, 2, 1, 2, 5, 2, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
    
    i = 0

    while i!=16:
        plt.annotate("v"+str(i), xy=(guardaPosicao[2*i], guardaPosicao[2*i+1]), xycoords='data', xytext=(guardaPosicao[2*i]+0.1,guardaPosicao[2*i+1]+0.1), textcoords='data')
        i = i + 1

def nomeiaPontos2():
    #lista com os pontos ordenados
    guardaPosicao = [0, 5, 2, 5, 0, 4, 1, 4, 3, 4, 1, 3, 3, 3, 0, 2, 3, 2, 4, 2, 5, 2, 0, 1, 2, 1, 5, 1, 1, 0, 5, 0]

    i = 0

    while i!=16:
        plt.annotate("v"+str(i), xy=(guardaPosicao[2*i], guardaPosicao[2*i+1]), xycoords='data', xytext=(guardaPosicao[2*i]+0.1,guardaPosicao[2*i+1]+0.1), textcoords='data')
        i = i + 1

def constroGráfico():
    plt.scatter(linhaPretaX, linhaPretaY, c='black')
    plt.plot(linhaPretaX, linhaPretaY, 'k')

    plt.scatter(linhaAzulX, linhaAzulY, c='blue')
    plt.plot(linhaAzulX, linhaAzulY, 'b')

    plt.scatter(linhaVermelhaX, linhaVermelhaY, c='red')
    plt.plot(linhaVermelhaX, linhaVermelhaY, 'r')

    plt.scatter(linhaVerdeX, linhaVerdeY, c='green')
    plt.plot(linhaVerdeX, linhaVerdeY, 'g')
    plt.grid(True)
    plt.axis([-0.5, 6.5, -0.5, 6.5])

def transformaMatrizEpsEmAdjacente(matrizEps):
    adjacente = np.zeros((matrizEps.shape[0], matrizEps.shape[0]))

    for i in range(matrizEps.shape[0]):
        inter1 = matrizEps[i][0]
        inter2 = matrizEps[i][1]

        adjacente[inter1][inter2] = 1
        adjacente[inter2][inter1] = 1

    return adjacente

#Exercicio 4.1

linhaPretaX    = [0, 0, 0, 0, 1]
linhaPretaY    = [5, 4, 3, 2, 2]
linhaAzulX     = [1, 1, 1, 2, 3]
linhaAzulY     = [2, 1, 0, 0, 0]
linhaVermelhaX = [1, 2, 3, 4, 5]
linhaVermelhaY = [1, 1, 1, 1, 1]
linhaVerdeX    = [4, 3, 5, 5, 5] 
linhaVerdeY    = [0, 0, 0, 1, 2]

plt.title('Grafo 1 Exercicio 4')
constroGráfico()
nomeiaPontos1()

plt.show()


linhaPretaX    = [0, 0, 1, 0, 0]
linhaPretaY    = [5, 4, 3, 2, 1]
linhaAzulX     = [0, 1, 2, 3, 3]
linhaAzulY     = [1, 0, 1, 2, 3]
linhaVermelhaX = [3, 4, 5, 5, 5]
linhaVermelhaY = [2, 2, 2, 1, 0]
linhaVerdeX    = [0, 1, 2, 3, 3]
linhaVerdeY    = [4, 4, 5, 4, 3]

plt.title('Grafo 2 Exercicio 4')
constroGráfico()
nomeiaPontos2()

plt.show()

#Exercicio 4.2 - Feito a partir dos gráficos gerados

#Exercicio 4.3

epsilon_1 = np.array([
    [0, 1],
    [1, 2], 
    [2, 3],
    [3, 4],
    [4, 6],
    [6, 7],
    [6, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 10],
    [10, 5],
    [10, 9],
    [9, 8],
    [8, 7]
])

#converte a matriz epsilon em adjacente
adjacente1 = transformaMatrizEpsEmAdjacente(epsilon_1)
#printa para verificar a igualdade entre a do relatório e a do programa
print(adjacente1)

# A matriz de adjacencia encontrada tem o autovalor com multiplicidade maior do que um, não é
# possivel fazer o calulo pela Metodo das potencias, potencias inversas ou do metodo QR.
# Assim, decidiu utilizar a funcao np.linalg.eig() para poder fazer o exercício

guardaValoresReais = np.linalg.eig(adjacente1)

autovalor1 = guardaValoresReais[0][0]
autovetor1 = guardaValoresReais[1][0]

print("Autovalor_G1: ", autovalor1)
print("Autovetor_G1: ", autovetor1)


#G2

epsilon_2 = np.array([
    [0, 2],
    [2, 3], 
    [2, 5],
    [5, 7],
    [7, 11],
    [11, 14],
    [14, 12],
    [12, 8],
    [8, 6],
    [6, 4],
    [4, 1],
    [1, 3],
    [8, 9],
    [9, 10],
    [10, 13],
    [13, 15]
])

#converte a matriz epsilon em adjacente
adjacente2 = transformaMatrizEpsEmAdjacente(epsilon_2)
#printa para verificar a igualdade entre a do relatório e a do programa
print(adjacente2)

guardaValoresReais = np.linalg.eig(adjacente2)

autovalor2 = guardaValoresReais[0][0]
autovetor2 = guardaValoresReais[1][0]

print("Autovalor_G2: ", autovalor2)
print("Autovetor_G2: ", autovetor2)

#Exercicio 4.4

#transforma o autovetor em unitario nao negativo
unitario1 = abs(autovetor1)/np.linalg.norm(autovetor1)

#encontra o maior autovalor e seu indice
for i in range(16):
    if unitario1[i]==max(unitario1):
        break

print("Vi com maior centralidade de G1 é: ", i)

#transforma o autovetor em unitario nao negativo
unitario2 = abs(autovetor2)/np.linalg.norm(autovetor2)

#encontra o maior autovalor e seu indice
for i in range(16):
    if unitario2[i]==max(unitario2):
        break

print("Vi com maior centralidade de G2 é: ", i)