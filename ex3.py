import numpy as np

# import matplotlib.pyplot as plt

EPS = 10**(-10)
ITR = 50

#retorna true se eh para continuar,
# ou seja, se algum elemento fora da diagonal principal eh maior que EPS
def criterioDeParada(matrizAk):
    for i in range(matrizAk.shape[0]):
        for j in range(matrizAk.shape[1]):
            if j < i and abs(matrizAk[i][j]) > EPS:
                return True
    return False

# faz a transformacao Householder, devolvendo a matriz correspondente
# note que para isso temos que fazer um tratamento no vetor passado a funcao
# paque que a multiplicacao de matrizes funcione como o exemplo abaixo               
# vetorV1 = [1,2,3,4] 1x4
# vtransp = [1]
#           [2]
#           [3]
#           [4] 4x1
#Hv = In -2(vetorV x vTransp)/(vetorV . vetorV)
def transfHouseholder(vetorV): 
    vTransp = vetorV.reshape((1, len(vetorV)))
    vetorV1 = vetorV.reshape((len(vetorV), 1))

    parcelaSuperior = np.dot(vetorV1, vTransp)
    parcelaInferior = np.inner(vetorV, vetorV)

    divisao = parcelaSuperior/parcelaInferior

    return np.identity(len(vetorV)) - 2 * divisao

# Calcula o vetorVi referente a coluna i da matrizA para ser usado na fatoracao householder
def calculaVetorVi (i, matrizA):
    vetorAi = np.copy(matrizA[i]) #recebe a coluna correspondente
    
    for k in range(i):
        vetorAi[k] = 0
        
    #cria o vetor da base canonica correspondente
    vetorEi = np.zeros(matrizA.shape[0])
    vetorEi[i] = 1

    # coloca o sinal no delta
    # se o valor vetorAi[i] for nulo, a matriz possui autovalores complexos
    if (vetorAi[i] != 0):
        delta = vetorAi[i] / abs(vetorAi[i])
    else:
        return False

    normaAi = np.linalg.norm(vetorAi)

    return vetorAi + delta*normaAi*vetorEi

def fatoracaoHouseholder(matrizAk):

    Q = np.identity((matrizAk.shape[0]))
    R = np.copy(matrizAk)

    #calcula calcula todos os Hvi, um para cada coluna, atualizando R e Q a cada iteracao
    for i in range(matrizAk.shape[1]-1): 

        vetorV = calculaVetorVi(i, matrizAk)

        # no caso de um autovalor ser complexo, paramos a fatoracao
        if type(vetorV) == type(False):
            return np.array([False, False], dtype=object)
            
        Hvi = transfHouseholder(vetorV)

        R = np.matmul(Hvi, R)
        Q = np.dot(Q, Hvi)
    return np.array([Q, R], dtype=object)


def encontraAutovaloresEAutovetores(matrizA):
    k = 1
    matrizAk_1= np.copy(matrizA)
    matrizVk = np.identity((matrizA.shape[1]))

    while k <= ITR and criterioDeParada(matrizAk_1): #calcula os Ai

        matrizAk = matrizAk_1
        Qk, Rk = fatoracaoHouseholder(matrizAk) 

        if type(Qk)==type(False) or type(Rk)==type(False):
            print("Existem autovalores complexos!\nFatoracao não pode ser feita")
            return 

        matrizAk_1 = np.dot(Rk, Qk)
        matrizVk = np.dot(matrizVk, Qk)
        k = k + 1

    return np.array( [matrizAk_1, matrizVk], dtype=object )

def encontraAutovalorDominante(matrizA):

    maiorValor = matrizA[0][0]

    for i in range(matrizA.shape[0]):
        if matrizA[i][i] > maiorValor:
            maiorValor = matrizA[i][i]

    return maiorValor


#Exercicio 3.1
print("\n\nExericio 3.1\n\n")

matrizA = np.array([
    [6, -2, -1],
    [-2, 6, -1],
    [-1, -1, 5]
])

resultado = encontraAutovaloresEAutovetores(matrizA)
print("Ak_1: ", resultado[0])
print("Vk: ", resultado[1])

# Exercicio 3.2

print("\n\nExericio 3.2\n\n")
matrizA = np.array([
        [1, 1],
        [-3, 1]    
])
resultado = encontraAutovaloresEAutovetores(matrizA)

# Exercicio 3.3

print("\n\nExericio 3.3\n\n")

matrizA = np.array([
        [3, -3],
        [0.33333, 5]    
])
resultado = encontraAutovaloresEAutovetores(matrizA)

print("Ak_1: ", resultado[0])
print("Vk: ", resultado[1])

# Exercicio 3.4.1
print("\n\nExericio 3.4.1\n\n")

#carrega a matriz utilizada no primeiro exercicio
matrizA = np.loadtxt("ex1_1.txt", dtype=float)

guardaValoresReais = np.linalg.eig(matrizA)
autovalorLambda1 = max(guardaValoresReais[0])

resultado = encontraAutovaloresEAutovetores(matrizA)

print("valor esperado: ", autovalorLambda1)
print("Ak_1: ", resultado[0])
print("Vk: ", resultado[1])

# exercicio 3.4.2
print("\n\nExericio 3.4.2\n\n")

#carrega a matriz utilizada no primeiro exercicio
matrizA = np.loadtxt("ex1_2.txt", dtype=float)

#calcula o autovalor real para testagem
guardaValoresReais = np.linalg.eig(matrizA)
autovalorLambda1 = max(guardaValoresReais[0])

resultado = encontraAutovaloresEAutovetores(matrizA)

print("valor esperado: 5", autovalorLambda1)
print("Ak_1: ", resultado[0])
print("Vk: ", resultado[1])