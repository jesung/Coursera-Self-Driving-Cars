import numpy as np
from numpy.linalg import inv

def predict_x(x, u, t_delta, F, G):
    return np.matmul(F, x) + np.matmul(G, u)

def predict_P(F, P, L, Q):
    return np.matmul(np.matmul(F, P),F.T) + np.matmul(np.matmul(L, Q),L.T)

def optimal_gain(P, H, M, R):
    return np.matmul(np.matmul(P, H.T),inv(np.matmul(H,np.matmul(P,H.T))+np.matmul(M,np.matmul(R,M.T))))

def main():
    #initialize variables
    x_cor = [[0],[5]]
    P_cor = [[0.01, 0],[0, 1]]

    t_delta = 0.5
    u = np.matrix([[-2]])
    y = np.matrix([[30]])
    S = 20
    D = 40
    
    F = np.matrix([[1, t_delta],[0, 1]])
    G = np.matrix([[0],[t_delta]])
    L = np.matrix([[1,0],[0,1]])
    M = np.matrix([[1]])
    Q = np.matrix([[0.1,0],[0,0.1]])
    R = np.matrix([[0.01]])
   

    #Compute predictions
    x_pred = predict_x(x_cor, u, t_delta, F, G)
    P_pred = predict_P(F, P_cor, L, Q)
        
    #Compute optimal gain
    H = np.matrix([[S/((D-x_pred[0,0])**2+S**2),0]])
    K = optimal_gain(P_pred, H, M, R)
       
    #Compute correction
    y_pred = np.degrees(np.arctan(S/(D-x_pred[0,0])))
    x_cor = x_pred + np.matmul(K,(y - y_pred))
    P_cor = np.multiply((1 - np.matmul(K,H)),P_pred)
    print(x_cor)
    print(P_pred)

if __name__ == "__main__":
    main()
