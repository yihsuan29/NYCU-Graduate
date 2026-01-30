import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--case', type = int, default=1, help='Input case')
    
    parser.add_argument('--h1', type = int, default=10, help='Number of first hidden units')
    parser.add_argument('--h2', type = int, default=10, help='Number of second hidden units')
    
    parser.add_argument('--a1', type = str, default="sigmoid", help='First activation function: sigmoid/ ReLU')
    parser.add_argument('--a2', type = str, default="sigmoid", help='Second activation function: sigmoid/ ReLU')
    
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epoch', type = int, default=15000, help='Number of training epochs')
    
    parser.add_argument('--a', type = int, default=1, help='Whether it has activation function.')
    parser.add_argument('--opt', type = int, default=0, help='Whether it use optimizer.')
    
    args = parser.parse_args()
    return args
    
    
def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        
        if 0.1*i==0.5:
            continue
        
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1) 

def evaluate(x,y,y_pred, size, loss, path):
    acc = 0
    for i in range(size):
        if y[i] == (y_pred[i]>=0.5):
            acc+=1
        print(f"Iter{i}|\t Ground truth: {y[i][0]}|\t Prediction:{y_pred[i][0]:.4f}")
    acc = acc*100/size
    print(f"loss={loss:.4f} accuracy={acc:.2f}%")
    with open(path, "a") as f:
        f.write(f"loss={loss:.4f} accuracy={acc:.2f}%\n")


def show_result(x,y,y_pred, path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if y_pred[i]<0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.savefig(path)
    #plt.show()
    
def draw_boudary(model, X,y, path):
    plt.figure(figsize=(6, 5))
    x_min, x_max = 0,1
    y_min, y_max = 0,1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    for j in range(Z.shape[0]):
        if Z[j][0] < 0.5:
            Z[j][0] = 0
        else:
            Z[j][0] = 1
            
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.4)
    
    for i in range(X.shape[0]):
        if y[i] < 0.5:
            plt.plot(X[i][0], X[i][1], 'ro')
        else:
            plt.plot(X[i][0], X[i][1], 'bo')
            
    plt.axis('off')
    plt.savefig(path)
    #plt.show()
    
    
def learning_curve(epochs, losses, path):
    plt.figure()
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(path)
    #plt.show()
    
    
def save_config(path, args):    
    with open(path, "w") as f:
        f.write(f"Activation = {args.a1}\n")
        f.write(f"h1 = {args.h1}\n")
        f.write(f"h2 = {args.h2}\n")
        f.write(f"lr = {args.lr}\n")
