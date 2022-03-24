import numpy as np

#Epsilon
Ep = 10**(-13)

#calcula o autovetor da proxima iteracao
def calculaProxAutoVetor (vetorA, vetorX):
    vetorMult = vetorA @ vetorX 
    norma = np.linalg.norm(vetorMult)
    return vetorMult/norma

#calcula o autovalor da proxima iteracao
def calculaProxAutoValor(vetorA, vetorX):
    parteSuperior = vetorX.T @ vetorA @ vetorX
    parteInferior = vetorX.T @ vetorX
    return parteSuperior/parteInferior

#gera um vetor de n linhas
def geraX_0(n):
    return np.random.random((n))

#seleciona o autovetor do autovalor dominante
def selecionaAutoVetor(A):
    eigenValues = np.linalg.eig(A)
    #seleciona o autovalor dominante
    lambda_n = np.max(eigenValues[0])
    i = 0
    for mi_k in eigenValues[0]:
        if mi_k == lambda_n:
            novaMatriz = eigenValues[1].T 
            return (novaMatriz[i,:]) 
        i = i + 1

#adaptacao do primeiro exercicio para encontrar o autovalor e autovetor
def encontraAutovalorEAutovetorDominante(matrizA, iteracoesMax):
    x_0 = geraX_0(matrizA.shape[0])
    x_0 = np.identity(matrizA.shape[0])
    autovetDominante = selecionaAutoVetor(matrizA)

    autovetor = x_0
    i = 1

    while (i<iteracoesMax and min(np.linalg.norm(autovetDominante+autovetor), np.linalg.norm(autovetDominante-autovetor))>Ep):
    
        autovalor = calculaProxAutoValor (matrizA, autovetor) 
        print(autovalor)
    
        i = i + 1

        autovetor = calculaProxAutoVetor (matrizA, autovetor)

    return np.array([autovalor, autovetor], dtype=object)

